import requests

# URL de la API
url = "http://localhost:8000/api/v1/extract-id"

# Ruta a la imagen (cambia si es diferente)
image_path = "./tests/test_images/1.webp"

# Enviar la solicitud
with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

# Mostrar el resultado
print("Status Code:", response.status_code)
print("Response:", response.json() if response.status_code == 200 else response.text)
