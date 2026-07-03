import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

def _save_processed_image(img: np.ndarray, step_name: str) -> Path:
    workspace_root = Path(__file__).resolve()
    output_dir = workspace_root.parent / "processed_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{step_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.png"
    output_path = output_dir / filename
    if not cv2.imwrite(str(output_path), img):
        raise Exception(f"No se pudo guardar la imagen procesada: {filename}")
    print(f"Saved: {filename}")
    return output_path

def test_crop_document(img_path: str):
    """
    Test the crop document function and save every step.
    """
    print(f"\n=== Testing with: {img_path} ===")
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return
    
    # Save original
    _save_processed_image(img, "original")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _save_processed_image(gray, "gray")

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _save_processed_image(blurred, "blurred")

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    _save_processed_image(edges, "edges")

    # Morphological operations to close gaps and connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated_edges = cv2.dilate(edges, kernel, iterations=3)
    _save_processed_image(dilated_edges, "dilated_edges")
    
    closed_edges = cv2.erode(dilated_edges, kernel, iterations=2)
    _save_processed_image(closed_edges, "closed_edges")

    # Find contours
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (assumed to be the document)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw contour on a copy for visualization
        contour_img = img.copy()
        cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 3)
        _save_processed_image(contour_img, "contour")

        # Add small padding (1%) to ensure we don't cut off content
        pad_x = int(w * 0.01)
        pad_y = int(h * 0.01)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.shape[1], x + w + pad_x)
        y2 = min(img.shape[0], y + h + pad_y)
        
        # Draw rectangle showing crop area
        crop_preview = img.copy()
        cv2.rectangle(crop_preview, (x1, y1), (x2, y2), (255, 0, 0), 3)
        _save_processed_image(crop_preview, "crop_area")

        # Only crop if the detected area is significantly smaller than the original
        if w * h < img.shape[0] * img.shape[1] * 0.9:
            cropped = img[y1:y2, x1:x2]
            _save_processed_image(cropped, "cropped")
            print(f"Cropped from {img.shape} to {cropped.shape}")
        else:
            print("No significant cropping needed")
            _save_processed_image(img, "final_no_crop")
    else:
        print("No contours found")
        _save_processed_image(img, "final_no_contours")

# Test with cedula images
test_images = [
    "/workspace/tests/test_images/cedula.jpg",
    "/workspace/tests/test_images/cedula2.jpg",
    "/workspace/tests/test_images/cedula3.jpg"
]

for img_path in test_images:
    test_crop_document(img_path)

print("\n=== All tests completed ===")
