#!/bin/bash

# ==============================================================================
# GENERAL SCRIPT FOR MASS 4D FLOW PIPELINE
# Runs the entire pipeline (DCM -> AI -> DCM) managing environments automatically
# ==============================================================================

# --- USER CONFIGURATION (adjust as needed) ---
ENV_NAME="4dflow_network_env"   # Name of the environment with required packages
SIF_IMAGE="tensorflow_2.6.1-gpu.sif"  # Name of the Singularity container file
CODE_DIR="code"                  # Relative path to the code folder



# 1. ARGUMENT CHECKING
if [ "$#" -ne 2 ]; then
    echo "❌ Incorrect usage."
    echo "Usage: bash run_pipeline.sh [DICOM_ORIGINAL_FOLDER] [ROOT_PATIENT_FOLDER]"
    echo "Example: bash run_pipeline.sh /path/to/00_dcmLR_006 /path/to/data/006"
    exit 1
fi

INPUT_DCM="$1"
OUTPUT_ROOT="$2"
PATIENT_ID=$(basename "$OUTPUT_ROOT")



echo "=========================================================="
echo " STARTING PIPELINE FOR PATIENT: $PATIENT_ID"
echo "=========================================================="

# 2. LOAD SYSTEM MODULES
# If running on a cluster that requires modules, uncomment the lines below and adjust as needed
echo "Loading system modules..."
#module load Anaconda 3 # or Miniforge/Miniconda or similar 
#module load Singularity # or Apptainer if needed             

# 3. INITIALIZE PYTHON ENVIRONMENT
# Try to source conda if available, otherwise assume environment is already active
if command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"    # environment previously created with required packages

else
    echo "⚠️ Warning: 'conda' command not found. Assuming the required environment is already active."
fi


# ---------------------------------------------------------
# STEP 1: PRE-PROCESSING (Use Conda)
# ---------------------------------------------------------
echo ""
echo ">>> [1/3] Converting DICOM to H5 (Local Python)"
python3 "$CODE_DIR/dcm_to_h5.py" "$INPUT_DCM" "$OUTPUT_ROOT" #add --force if needed

if [ $? -ne 0 ]; then
    echo "❌ Error on Step 1. Aborting."
    exit 1
fi

# Path to the generated low-res H5 file
H5_INPUT="$OUTPUT_ROOT/01_h5_input/${PATIENT_ID}_LowRes.h5"

# ---------------------------------------------------------
# STEP 2: SUPER-RESOLUTION AI (Use Singularity)
# ---------------------------------------------------------
echo ""
echo ">>> [2/3] Executing Super-Resolution AI (Environment: Singularity)"

# Check if Singularity is available
if command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
elif command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
else
    echo "❌ Error: Singularity command not found. Please ensure Singularity is installed and accessible."
    exit 1
fi

# Execute container 
$CONTAINER_CMD run --nv "$SIF_IMAGE" python3 "$CODE_DIR/src/predictor_temporal_invivo.py" "$H5_INPUT" "$OUTPUT_ROOT"

if [ $? -ne 0 ]; then
    echo "❌ Error on Step 2. Aborting."
    exit 1
fi

# Search for the predicted H5 file
H5_PREDICTED=$(find "$OUTPUT_ROOT/02_h5_output" -name "*.h5" | head -n 1)

if [ -z "$H5_PREDICTED" ]; then
    echo "❌ Error: Did not find the AI output file."
    exit 1
fi

# ---------------------------------------------------------
# STEP 3: POST-PROCESSING (Use Conda)
# ---------------------------------------------------------
echo ""
echo ">>> [3/3] Reconstructing final DICOMs (Local Python)..."

# Define the template directory (Sorted LowRes from Step 1)
TEMPLATE_DIR="$OUTPUT_ROOT/01_h5_input"

python3 "$CODE_DIR/h5_to_dcm.py" "$H5_PREDICTED" "$TEMPLATE_DIR" "$OUTPUT_ROOT"

if [ $? -ne 0 ]; then
    echo "❌ Error on Step 3. Aborting."
    exit 1
fi

echo ""
echo "=========================================================="
echo "PROCESS COMPLETED SUCCESSFULLY!"
echo "Results in: $OUTPUT_ROOT/03_dicom_final"
echo "=========================================================="