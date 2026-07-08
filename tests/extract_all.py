import requests
import os

# URL de la API
url = "http://localhost:8000/api/v1/extract-id"

# Extensiones de imagen válidas
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp', '.pdf')

# Archivo de salida
output_file = "results.txt"

# Lista de imágenes en test_images
test_images_dir = "./tests/test_images"
if not os.path.exists(test_images_dir):
    print(f"Directorio {test_images_dir} no encontrado.")
    exit(1)

image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(image_extensions)]

if not image_files:
    print("No se encontraron imágenes en test_images.")
    exit(1)

# Abrir archivo de salida
with open(output_file, "w", encoding="utf-8") as outfile:
    for filename in image_files:
        image_path = os.path.join(test_images_dir, filename)
        try:
            with open(image_path, "rb") as f:
                files = {"file": f}
                response = requests.post(url, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                outfile.write(f"{filename}: {result}\n\n")
                print(f"Procesado: {filename}")
            else:
                outfile.write(f"{filename}: Error {response.status_code} - {response.text}\n")
                print(f"Error en {filename}: {response.status_code}")
        except Exception as e:
            outfile.write(f"{filename}: Excepción - {str(e)}\n")
            print(f"Excepción en {filename}: {str(e)}")

print(f"Resultados guardados en {output_file}")