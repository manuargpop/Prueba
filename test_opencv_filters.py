import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


def save_processed_image(img: np.ndarray, prefix: str, original_filename: str) -> Path:
    """Save processed image with descriptive filename."""
    workspace_root = Path(__file__).resolve().parent
    output_dir = workspace_root / "processed_images" / "opencv_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with original name and processing type
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_original = Path(original_filename).stem.replace(" ", "_")
    filename = f"{timestamp}_{safe_original}_{prefix}.png"
    output_path = output_dir / filename
    
    if not cv2.imwrite(str(output_path), img):
        raise Exception(f"No se pudo guardar la imagen: {filename}")
    
    return output_path


def apply_laplacian(img: np.ndarray) -> np.ndarray:
    """Apply Laplacian operator for edge enhancement."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise before Laplacian
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Apply Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    # Convert to uint8 (absolute values)
    laplacian_uint8 = np.uint8(np.absolute(laplacian))
    # Convert back to BGR for saving
    return cv2.cvtColor(laplacian_uint8, cv2.COLOR_GRAY2BGR)


def apply_canny_edge(img: np.ndarray) -> np.ndarray:
    """Apply Canny edge detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    # Convert to BGR for saving
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def apply_sobel(img: np.ndarray) -> np.ndarray:
    """Apply Sobel edge detection (combines X and Y gradients)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Apply Sobel in X and Y directions
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    # Combine both gradients
    sobel_combined = cv2.addWeighted(
        np.uint8(np.absolute(sobel_x)), 0.5,
        np.uint8(np.absolute(sobel_y)), 0.5,
        0
    )
    # Convert to BGR for saving
    return cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)


def process_test_images():
    """Process all test images with different OpenCV functions."""
    test_images_dir = Path(__file__).resolve().parent / "tests" / "test_images"
    
    if not test_images_dir.exists():
        print(f"❌ Directorio de imágenes de prueba no encontrado: {test_images_dir}")
        return
    
    # Get all image files
    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    image_files = [
        f for f in test_images_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print("❌ No se encontraron imágenes en el directorio de prueba")
        return
    
    print(f"📁 Encontradas {len(image_files)} imágenes para procesar")
    print("=" * 60)
    
    processed_count = 0
    
    for img_path in image_files:
        print(f"\n🔄 Procesando: {img_path.name}")
        
        # Read original image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"   ⚠️  No se pudo leer la imagen: {img_path.name}")
            continue
        
        # Save original image
        original_path = save_processed_image(img.copy(), "original", img_path.name)
        print(f"   ✓ Original guardado: {original_path.name}")
        processed_count += 1
        
        # Apply and save Laplacian
        try:
            laplacian_img = apply_laplacian(img)
            laplacian_path = save_processed_image(laplacian_img, "laplacian", img_path.name)
            print(f"   ✓ Laplacian guardado: {laplacian_path.name}")
            processed_count += 1
        except Exception as e:
            print(f"   ❌ Error en Laplacian: {e}")
        
        # Apply and save Canny Edge
        try:
            canny_img = apply_canny_edge(img)
            canny_path = save_processed_image(canny_img, "canny_edge", img_path.name)
            print(f"   ✓ Canny Edge guardado: {canny_path.name}")
            processed_count += 1
        except Exception as e:
            print(f"   ❌ Error en Canny Edge: {e}")
        
        # Apply and save Sobel
        try:
            sobel_img = apply_sobel(img)
            sobel_path = save_processed_image(sobel_img, "sobel", img_path.name)
            print(f"   ✓ Sobel guardado: {sobel_path.name}")
            processed_count += 1
        except Exception as e:
            print(f"   ❌ Error en Sobel: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ Proceso completado!")
    print(f"   Imágenes procesadas: {processed_count}")
    print(f"   Directorio de salida: {Path(__file__).resolve().parent / 'processed_images' / 'opencv_tests'}")
    print("\n💡 Revisa las imágenes generadas para comparar los resultados de cada método.")


if __name__ == "__main__":
    process_test_images()
