#!/usr/bin/env python3
"""
H5 to DICOM Reconstructor
-------------------------
Converts High-Resolution H5 data (AI output) back into DICOM format.
It uses the original Low-Resolution DICOMs as templates for headers/metadata
and interpolates the Magnitude image to match the new number of frames.
"""

import h5py
import pydicom
import numpy as np
import os
import glob
import logging
import argparse
import sys
import re
from pydicom.uid import generate_uid
from tqdm import tqdm
import scipy.ndimage

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RECONSTRUCTION_RULES = {
    'u': {
        'candidates': ['u_combined', 'u', 'vel_u'],
        'template': 'RL',
        'pretty_name': 'RL',
        'type': 'phase'
    },
    'v': {
        'candidates': ['v_combined', 'v', 'vel_v'],
        'template': 'AP',
        'pretty_name': 'AP',
        'type': 'phase'
    },
    'w': {
        'candidates': ['w_combined', 'w', 'vel_w'],
        'template': 'FH',
        'pretty_name': 'FH',
        'type': 'phase'
    },
    'mag': {
        'candidates': ['mag_w', 'mag', 'magnitude'],
        'template': 'MAG',
        'pretty_name': 'MAG',
        'type': 'magnitude'
    }
}

def get_venc(sequence_name):
    "Extract VENC from SequenceName, default to 150.0 if not found."
    if not sequence_name:
        return 150.0
    match = re.search(r"v(\d+)", sequence_name, re.IGNORECASE)
    return float(match.group(1)) if match else 150.0

def convert_velocity_to_phase_pixels(ds, velocity_data_2d, venc):
    "Convert velocity data (cm/s) to DICOM phase pixel representation (0-4095)."
    "Pixel value = ((velocity / VENC) * 4095 + 4096) / 2"
    phaseRange = 4095.0
    if venc == 0: 
        venc = 150.0
    
    scaled_phase_data = (velocity_data_2d / venc) * phaseRange
    raw_pixel_data = (scaled_phase_data + 4096.0) / 2.0
    
    int_pixel_data = np.rint(raw_pixel_data).astype(np.uint16)
    int_pixel_data = np.clip(int_pixel_data, 0, 4095)
    
    ds.PixelData = int_pixel_data.tobytes()
    ds.PixelRepresentation = 0
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.SmallestImagePixelValue = int(np.min(int_pixel_data))
    ds.LargestImagePixelValue = int(np.max(int_pixel_data))
    return ds

def convert_magnitude_pixels(ds, mag_data_2d):
    "Convert magnitude data to DICOM pixel representation (0-4095)"
    max_val = np.max(mag_data_2d) if np.max(mag_data_2d) > 0 else 1.0
    
    scale = 4095.0 / max_val
    int_pixel_data = np.rint(mag_data_2d * scale).astype(np.uint16)
    int_pixel_data = np.clip(int_pixel_data, 0, 4095)

    ds.PixelData = int_pixel_data.tobytes()
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1.0 / scale
    ds.RescaleIntercept = 0.0
    return ds

def load_and_interpolate_magnitude(template_dir, target_phases, num_slices, rows, cols):
    """
    Loads original magnitude DICOMs and interpolates them in the time dimension
    to match the high temporal resolution of the AI output.
    """
    logger.info(f"Searching magnitude templates in: {template_dir}")
    all_files = sorted(glob.glob(os.path.join(template_dir, "*")))
    valid_files = []

    for f in all_files:
        if os.path.basename(f).startswith('.'):
            continue
        try:
            pydicom.dcmread(f, stop_before_pixels=True)
            valid_files.append(f)
        except:
            pass

    if not valid_files:
        raise ValueError(f"No magnitude template files found in {template_dir}")
        return None
    
    logger.info(f"Found {len(valid_files)} valid magnitude DICOM files.")
    orig_phases = len(valid_files) // num_slices
    if orig_phases == 0:
        return None
    
    logger.info(f"Interpolating Magnitude: {orig_phases} -> {target_phases} phases")
    vol_orig = np.zeros((rows, cols, num_slices, orig_phases), dtype=np.float32)

    for i, fpath in enumerate(valid_files):
        s = i // orig_phases
        p = i % orig_phases
        if s >= num_slices: break
        try:
            ds = pydicom.dcmread(fpath)
            px = ds.pixel_array.astype(np.float32)
            if px.shape != (rows, cols):
                px = scipy.ndimage.zoom(px, (rows / px.shape[0], cols / px.shape[1]), order=1)
            # Apply rescale slope/intercept to get real values
            slope = getattr(ds, 'RescaleSlope', 1.0)
            intercept = getattr(ds, 'RescaleIntercept', 0.0)
            vol_orig[:, :, s, p] = px * slope + intercept
        except Exception: 
            pass
    
    zoom_factors = (1, 1, 1, target_phases / orig_phases)
    return scipy.ndimage.zoom(vol_orig, zoom_factors, order=1)

def robuts_transpose(data, expected_slices):
    """
    Identifies dimensions based on expected slice count.
    Target format: (rows, cols, slices, time)
    """
    shape = data.shape
    ndim = len(shape)

    if ndim == 4:
        matches = [i for i, s in enumerate(shape) if s == expected_slices]

        if not matches:
            logger.warning(f"Could not find dimension matching expected slices {expected_slices} in shape {shape}.")
            logger.warning("Assuming  standard format (Time, X, Y, Z).")
            return np.transpose(data, (1, 2, 3, 0))
        
        if matches[-1] == 3:
            return np.transpose(data, (1, 2, 3, 0))
        if matches[0] == 0:
            return np.transpose(data, (1, 2, 0, 3))
    
    # Fallback
    return np.transpose(data, (1, 2, 3, 0))

def count_original_slices(template_dir_path):
    "Counts expected slices by looking at the any folder in templates"

    subfolders = [f.path for f in os.scandir(template_dir_path) if f.is_dir()]
    valid_files = []

    # Find the first folder with files
    for folder in subfolders:
        files = glob.glob(os.path.join(folder, "*.dcm"))
        if len(files) > 0:
            valid_files = sorted(files)
            break
    
    if not valid_files:
        logger.warning(f"No DICOM templates found to calculate slices. Defaulting to 48")
        return 48
    
    try:
        ds = pydicom.dcmread(valid_files[0])
        phases = getattr(ds, 'NumberOfTemporalPositions', 0)
        if phases == 0:
            phases = getattr(ds, 'CardiacNumberOfImages', 0)
        
        phases = int(phases)
        if phases == 0:
            phases = 20 # Default fallback
    except Exception as e:
        logger.warning(f"Could not read template tags: {e}. Defaulting phases to 1")
        phases = 1

    num_slices = len(valid_files) // phases
    return max(1, num_slices)

def reconstruct_dicoms_from_h5(h5_file_path, template_dir_path, output_dir_path):
    "Reconstruct DICOM files from H5 data using templates."
    expected_slices = count_original_slices(template_dir_path)
    logger.info(f"Detected original geometry: {expected_slices} slices.")
    try:
        logger.info(f"Opening H5: {h5_file_path}")
        h5_data = {}
        found_shape = None

        with h5py.File(h5_file_path, 'r') as f:
            h5_data = {}
            for target_name, rule in RECONSTRUCTION_RULES.items():
                found = False
                for candidate in rule['candidates']:
                    if candidate in f:
                        raw_data = f[candidate][:]
                        data = robuts_transpose(raw_data, expected_slices)
                        h5_data[target_name] = data

                        if found_shape is None:
                            found_shape = data.shape
                        found = True
                        break
                if not found and target_name != 'mag':
                    logger.warning(f"Data for '{target_name}' not found in H5.")
            
            if found_shape is None:
                logger.error("No valid data found in H5 file.")
                return
            
            rows, cols, num_slices, num_phases = found_shape
            logger.info(f"Data shape: {found_shape} (rows, cols, slices, phases). Total Frames: {num_phases * num_slices}")

            if num_slices != expected_slices:
                logger.warning(f"Expected slices ({expected_slices}) does not match H5 data slices ({num_slices}).")

    except Exception as e:
        logger.error(f"Error reading H5: {e}")
        return
    
    # Magnitude interpolations from original DICOMs
    mag_dir = os.path.join(template_dir_path, 'MAG')
    if os.path.exists(mag_dir):
        interp_mag = load_and_interpolate_magnitude(mag_dir, num_phases, num_slices, rows, cols)
        if interp_mag is not None:
            h5_data['mag'] = interp_mag
            logger.info("Magnitude successfully interpolated.")
    else:
        logger.warning(f"Magnitude template dir not found at {mag_dir}. Skipping MAG.")

    for target_key, data_array in h5_data.items():
        rule = RECONSTRUCTION_RULES.get(target_key)
        if not rule: continue

        template_folder_name = rule['template']
        pretty_name = rule['pretty_name']
        recon_type = rule['type']

        logger.info(f"--- Generating Series: {template_folder_name} ---")

        curr_temp_path = os.path.join(template_dir_path, template_folder_name)
        templates = sorted(glob.glob(os.path.join(curr_temp_path, "*")))
        templates = [t for t in templates if os.path.isfile(t) and "DICOMDIR" not in t and not os.path.basename(t).startswith('.')]

        if not templates:
            logger.error(f"No templates found in {curr_temp_path}")
            continue

        num_templates = len(templates)

        # Prepare output directory
        current_output_dir = os.path.join(output_dir_path, template_folder_name)
        os.makedirs(current_output_dir, exist_ok=True)


        # Detect RR Interval
        try:
            ds_temp = pydicom.dcmread(templates[0])
            if ds_temp.get('NominalInterval'):
                rr_interval = float(ds_temp.NominalInterval)
                logger.info(f"Detected RR interval: {rr_interval} ms")
            else:
                rr_interval = 1000.0 # Default to 1 second
                logger.warning("NominalInterval not found. Using 1000ms by default.")
        except Exception as e:
            logger.warning(f"Error reading time metadata: {e}")
            rr_interval = 1000.0

        count = 0
        for s in range(num_slices):
            slice_idx = (num_slices - 1) - s # if it's needed to flip slices (change to s if not)
            
            for p in range(num_phases):
                # Selection of original template (Nearest Neighbor)
                orig_phases_est = num_templates // num_slices
                if orig_phases_est > 0:
                    p_orig = int((p / num_phases) * orig_phases_est)
                    template_idx = (s * orig_phases_est) + p_orig
                    if template_idx >= len(templates): 
                        template_idx = len(templates) - 1
                else:
                    template_idx = 0 # Fallback     

                try:
                    ds = pydicom.dcmread(templates[template_idx])
                    pixel_data_2d = data_array[:, :, slice_idx, p]

                    if recon_type == 'phase':
                        venc = get_venc(ds.SequenceName)
                        ds = convert_velocity_to_phase_pixels(ds, pixel_data_2d, venc)
                    else: 
                        ds = convert_magnitude_pixels(ds, pixel_data_2d)
                    
                    # Update metadata
                    ds.SOPInstanceUID = generate_uid()
                    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
                    ds.InstanceNumber = count + 1
                    
                    # Time Correction
                    new_trigger_time = (p / num_phases) * rr_interval
                    ds.TriggerTime = new_trigger_time
                    ds.CardiacNumberOfImages = num_phases

                    
                    fname = f"{pretty_name}_sl{s:04d}_ph{p:04d}_SR.dcm"
                    ds.save_as(os.path.join(current_output_dir, fname))
                    count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing slice {s} phase {p}: {e}")

        logger.info(f"Saved {count} files in {template_folder_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_file", help="Path to the input .h5 file", type=str)
    parser.add_argument("template_dir", help="Directory containing original sorted DICOM templates", type=str)
    parser.add_argument("output_dir", help="Root directory for output",type=str)
    args = parser.parse_args()

    if not os.path.isfile(args.h5_file):
        print(f"❌ ERROR: The H5 file '{args.h5_file}' does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(args.template_dir):
        print(f"❌ ERROR: The template directory '{args.template_dir}' does not exist.")
        sys.exit(1)
    
    root_path = os.path.normpath(args.output_dir)
    pacient_id = os.path.basename(root_path)
    
    new_folder_name = "03_dicom_final"
    final_output_dir = os.path.join(root_path, new_folder_name)
    print(f"Output configured to: {final_output_dir}")
    
    if not os.path.isdir(final_output_dir):
        os.makedirs(final_output_dir, exist_ok=True)
    
    reconstruct_dicoms_from_h5(args.h5_file, args.template_dir, final_output_dir)