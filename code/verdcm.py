import pydicom
import sys


# CAMBIA ESTO por la ruta a tu archivo .dcm
#FILE_PATH = '/proj/multipress/users/x_raqri/MASS_export/dcm_reconstruido/U_Velocity/SER_0073_sl0047_ph0010.dcm'
FILE_PATH = '/proj/multipress/users/x_raqri/MASS_export/006_h5/4D_WholeHeart_3mm/AP/SER_0084_sl0001_ph0014.dcm'

try:
    # Lee el archivo DICOM
    ds = pydicom.dcmread(FILE_PATH)
    
    print(f"--- 🩺 Metadatos del archivo DICOM: {FILE_PATH} ---\n")
    
    # Imprimir el objeto 'ds' (Dataset) muestra todos los metadatos
    print(ds)

except FileNotFoundError:
    print(f"Error: No se pudo encontrar el archivo en la ruta '{FILE_PATH}'")
except pydicom.errors.InvalidDicomError:
    print(f"Error: El archivo '{FILE_PATH}' no es un archivo DICOM válido o está corrupto.")
except Exception as e:
    print(f"Ha ocurrido un error inesperado: {e}")