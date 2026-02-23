import h5py
import sys

def print_hdf5_metadata(name, obj):
    """
    Función 'callback' para imprimir la estructura y atributos de un objeto HDF5.
    Se usa con f.visititems().
    """
    
    # Identifica el tipo de objeto (Grupo o Dataset)
    if isinstance(obj, h5py.Group):
        print(f"📁 Grupo: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"📄 Dataset: {name}")
        print(f"   Shape: {obj.shape}")
        print(f"   Dtype: {obj.dtype}")
    else:
        print(f"❓ Objeto desconocido: {name} (Tipo: {type(obj)})")

    # Imprime los atributos (metadatos) si existen
    if obj.attrs:
        print("   --- Metadatos (Atributos) ---")
        for key, val in obj.attrs.items():
            # Decodifica si es bytestring (común en HDF5)
            if isinstance(val, (bytes, bytearray)):
                val = val.decode('utf-8', 'ignore')
            print(f"     -> {key}: {val}")
        print("   -----------------------------")
    
    print("-" * 30) # Separador

# --- SCRIPT PRINCIPAL ---

# CAMBIA ESTO por la ruta a tu archivo .h5
FILE_PATH = '/Users/raque/export_MASS/export/4D_WholeHeart_3mm/4D_WholeHeart_3mm.h5'

try:
    # Abre el archivo en modo lectura ('r')
    with h5py.File(FILE_PATH, 'r') as f:
        print(f"Inspeccionando archivo: {FILE_PATH}\n")
        
        # --- INICIO DE CÓDIGO NUEVO ---
        print("--- 🔬 Metadatos en la RAÍZ (/) del archivo ---")
        if f.attrs:
            for key, val in f.attrs.items():
                if isinstance(val, (bytes, bytearray)):
                    val = val.decode('utf-8', 'ignore')
                # Imprimir el atributo de la raíz
                print(f"  -> {key}: {val}")
        else:
            print("     (No se encontraron atributos en la raíz)")
        print("-" * 50)
        # --- FIN DE CÓDIGO NUEVO ---

        print("\n--- 📂 Contenido del archivo (Datasets/Grupos) ---")
        # 'visititems' recorre cada objeto (grupo y dataset) en el archivo
        # y llama a la función 'print_hdf5_metadata' para cada uno.
        f.visititems(print_hdf5_metadata)

except FileNotFoundError:
    print(f"Error: No se pudo encontrar el archivo en la ruta '{FILE_PATH}'")
except OSError:
    print(f"Error: No se pudo abrir el archivo '{FILE_PATH}'.")
    print("Verifica que sea un archivo HDF5 válido y no esté corrupto.")
except Exception as e:
    print(f"Ha ocurrido un error inesperado: {e}")