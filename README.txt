==============================================================================
USER MANUAL: MASS 4D FLOW SR NETWORK PROCESSING PIPELINE
==============================================================================

PROJECT OVERVIEW
------------------
This pipeline improves the Temporal Resolution of 4D Flow images using Deep Learning. 
The workflow consists of three stages:

   1. Pre-processing: Conversion of original DICOMs to H5 format
   2. AI Prediction: Temporal Super-Resolution using 4DFlowNet (via Singularity)
   3. Post-processing: Reconstruction of predicted data (H5) back to DICOM format

------------------------------------------------------------------------------
A. SYSTEM REQUIREMENTS AND INSTALLATION
------------------------------------------------------------------------------

To run this pipeline you need Singularity (Apptainer) installed on your system and a specific Python environment.
The pipeline has been validated using NVIDIA A100 GPUs on a HPC Cluster.

* HARDWARE:
   - Step 1 & 3 (Data conversion): Can run on standard CPUs (Local PC/Mac/Linux)
   - Step 2 (Temporal Super-Resolution Network): Requires an NVIDIA GPU with CUDA support.


1. PYTHON DEPENDENCIES
   The script require Python 3.10. 
   You can install the dependencies in any environment manager (Conda, Mamba or Python venv) using pip.

   Required packages:
   - numpy==2.2.6
   - scipy==1.15.2
   - matplotlib==3.10.7
   - h5py
   - pydicom
   - tqdm
   - nibabel

2. INSTALLATION EXAMPLE (Using Conda)
   Run the following commands in your terminal to set up the enironment:

   $ conda create -n 4dflow_network_env python=3.10
   $ conda activate 4dflow_network_env
   $ pip install numpy==2.2.6 h5py pydicom scipy==1.15.2 tqdm nibabel matplotlib==3.10.7

   * NOTE: If you use OTHER environment, remember to update the activation command on "run_pipeline.sh"

3. SINGULARITY IMAGE
   Ensure the container file is located in the project root directory:
   'tensorflow_2.6.1-gpu.sif'


------------------------------------------------------------------------------
B. AUTOMATED EXECUTION
------------------------------------------------------------------------------

The easiest way to run the pipeline is using the master script "run_pipeline.sh".
This script handles environment activation and sequential execution of all steps.

USAGE:
   ./run_pipeline.sh [INPUT_DICOM_DIR] [OUTPUT_ROOT_DIR]

   ARGUMENTS:
      - [INPUT_DICOM_DIR]: Path to the folder containing the original DICOMs.
      - [OUTPUT_ROOT_DIR]: Path where the patient's data structure will be created.

REAL EXAMPLE:
   ./run_pipeline.sh /home/user/data/Patient_006/00_dicom_raw /home/user/data/Patient_006


------------------------------------------------------------------------------
C. MANUAL EXECUTION (STEP BY STEP)
------------------------------------------------------------------------------

Use this section if you need to debug or run specific steps individually.
Ensure your Python environment is active: 
   ($ conda activate 4dflow_network_env) #in our example


➡️ >>> STEP 1: DICOM TO H5 (Pre-processing)
----------------------------------------
Scans for DICOMs, sorts flow components (RL/AP/FH/MAG) and generates a .h5 file.

   COMMAND:
   python3 code/dcm_to_h5.py [ORIGINAL_DICOM_FOLDER] [OUTPUT_ROOT_DIR]
   
   * Output: Creates folder '01_h5_input' containing the Low-Res H5 file and the original DICOMs sorted.


➡️ >>> STEP 2: AI PREDICTION (Super-Resolution)
--------------------------------------------
   This step MUST be run with Singularity to access the GPU and TensorFlow.
   The script includes a patch for Keras version compatibility.

   COMMANDS:
   singularity run --nv tensorflow_2.6.1-gpu.sif 
   python3 code/src/predictor_temporal_invivo.py [INPUT_H5_FILE] [OUTPUT_ROOT_DIR]

   * Input Note: [INPUT_H5_FILE] is the file generated in Step 1
   * Output: Creates folder '02_h5_output' containing the High-Res H5 file.


➡️ >>> STEP 3: H5 TO DICOM (Post-processing)
-----------------------------------------
Converts the AI prediction (H5) back to DICOM format using the headers 
of the original files as templates.

   COMMAND:
   python3 code/h5_to_dcm.py [PREDICTED_H5_FILE] [ORIGINAL_TEMPLATE_DIR] [OUTPUT_ROOT_DIR]

   * Important: [TEMPLATE_DIR] should point to the '01_h5_input' folder generated in Step 1 
   (where sorted templates are stored), NOT the raw input folder.
   * Output: Creates folder '03_dicom_final' containing the final DICOM files.


------------------------------------------------------------------------------
D. FINAL RESULTS & EXPORT
------------------------------------------------------------------------------

The final High-Resolution DICOMs will be located in:
   [OUTPUT_ROOT_DIR]/03_dicom_final

To download the results easily, zip the folder:
   zip -r results_[patientID].zip 03_dicom_final
