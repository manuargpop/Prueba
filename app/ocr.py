import cv2
import numpy as np
from fastapi import HTTPException
import easyocr
from pathlib import Path
from datetime import datetime


reader = easyocr.Reader(["es"], gpu=False, verbose=False)


def _save_processed_image(img: np.ndarray) -> Path:
    workspace_root = Path(__file__).resolve().parents[1]
    output_dir = workspace_root / "processed_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = datetime.utcnow().strftime("processed_%Y%m%d_%H%M%S_%f.png")
    output_path = output_dir / filename
    if not cv2.imwrite(str(output_path), img):
        raise HTTPException(500, "No se pudo guardar la imagen procesada")
    return output_path

def _crop_document(img: np.ndarray) -> np.ndarray:
    """
    Crop the image to remove background and keep only the document.
    Uses edge detection and contour analysis to find the document boundaries.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _save_processed_image(blurred)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    _save_processed_image(edges)

    # Morphological operations to close gaps and connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated_edges = cv2.dilate(edges, kernel, iterations=3)
    _save_processed_image(dilated_edges)
    
    closed_edges = cv2.erode(dilated_edges, kernel, iterations=2)
    _save_processed_image(closed_edges)

    # Find contours
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (assumed to be the document)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add small padding (1%) to ensure we don't cut off content
        pad_x = int(w * 0.01)
        pad_y = int(h * 0.01)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.shape[1], x + w + pad_x)
        y2 = min(img.shape[0], y + h + pad_y)

        # Only crop if the detected area is significantly smaller than the original
        # This prevents unnecessary cropping when no background is detected
        if w * h < img.shape[0] * img.shape[1] * 0.9:
            cropped = img[y1:y2, x1:x2]
            _save_processed_image(cropped)
            return cropped

    # Return original image if no significant contour found
    return img

def preprocess_image(img_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Imagen inválida o corrupta")
    
    # Crop out background from the document photo
    img = _crop_document(img)
    	
    height, width = img.shape[:2]
    # Hide 1/4 (25%) of the right side of the image
    remove_x = int(width * 0.75)
    remove_y = 0
    remove_w = int(width * 0.25)
    remove_h = height
    
    processed = img.copy()
    cv2.rectangle(processed, (remove_x, remove_y), (width, remove_h), (255, 255, 255), thickness=-1)
    
    _save_processed_image(processed)
    return processed


def extract_raw_text(img_bytes: bytes) -> str:
    processed = preprocess_image(img_bytes)
    ocr_result = reader.readtext(processed, detail=1, paragraph=False)
    if not ocr_result:
        raise HTTPException(422, "No se detectó texto en la imagen")
    return " ".join([line[1] for line in ocr_result])
