import cv2
import numpy as np
from fastapi import HTTPException
# import easyocr
from pathlib import Path
from datetime import datetime
from pdf2image import convert_from_bytes
import base64
from io import BytesIO

# reader = easyocr.Reader(["es"], gpu=False, verbose=False)

def _convert_pdf_to_image(img_bytes: bytes) -> bytes:
    """
    Convert PDF bytes to PNG image bytes.
    Uses pdf2image with Poppler to render the first page at 300 DPI.
    Returns PNG format for optimal OCR quality.
    """
    try:
        # Convert first page of PDF to PIL Image at 300 DPI (optimal for OCR)
        images = convert_from_bytes(img_bytes, dpi=300, first_page=1, last_page=1)

        if not images:
            raise HTTPException(400, "PDF vacío o inválido")

        # Get first page only (we process one document at a time)
        pil_image = images[0]

        # Convert PIL Image to bytes in PNG format (lossless, best for OCR)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()

    except Exception as e:
        raise HTTPException(400, f"Error al procesar PDF: {str(e)}")


def _is_pdf_content(img_bytes: bytes) -> bool:
    """Check if the content is a PDF by looking at the magic bytes."""
    return img_bytes[:4] == b'%PDF'

# esta funcion es solo para debugg, se comenta los llamados a esta para el pase a produccion
def _save_processed_image_step(img: np.ndarray, step_name: str) -> Path:
    """Save intermediate processing steps for debugging."""
    workspace_root = Path(__file__).resolve().parents[1]
    output_dir = workspace_root / "processed_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{step_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.png"
    output_path = output_dir / filename
    if not cv2.imwrite(str(output_path), img):
        raise HTTPException(500, "No se pudo guardar la imagen procesada")
    return output_path


def _save_processed_image(img: np.ndarray) -> Path:
    workspace_root = Path(__file__).resolve().parents[1]
    output_dir = workspace_root / "processed_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = datetime.utcnow().strftime("processed_%Y%m%d_%H%M%S_%f.png")
    output_path = output_dir / filename
    if not cv2.imwrite(str(output_path), img):
        raise HTTPException(500, "No se pudo guardar la imagen procesada")
    return output_path

def _downscale_high_quality_image(img: np.ndarray, max_width: int = 2000, max_height: int = 1500) -> np.ndarray:
    """
    Downscale image if it's too high quality to avoid detecting noise/small pixels in background.
    Only processes if image exceeds maximum dimensions.
    """
    height, width = img.shape[:2]

    # Check if image exceeds maximum dimensions
    if width <= max_width and height <= max_height:
        return img

    # Calculate scaling factor to fit within max dimensions while preserving aspect ratio
    scale = min(max_width / width, max_height / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize using INTER_AREA which is best for shrinking images
    downscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Save the downscaled image for debugging
    # _save_processed_image_step(downscaled, "00_downscaled")

    return downscaled


def _crop_document(img: np.ndarray) -> np.ndarray:
    """
    Crop the image to remove background and keep only the document.
    Uses edge detection and contour analysis to find the document boundaries.
    """
    # First, downscale if image is too high quality to avoid noise detection
    img = _downscale_high_quality_image(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _save_processed_image_step(gray, "01_grayscale")

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _save_processed_image_step(blurred, "02_gaussian_blur")

    # Edge detection
    edges = cv2.Canny(blurred, 50, 100, 5)
    _save_processed_image_step(edges, "03_canny_edges")

    # Morphological operations to close gaps and connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated_edges = cv2.dilate(edges, kernel, iterations=3)
    _save_processed_image_step(dilated_edges, "04_dilated_edges")

    closed_edges = cv2.erode(dilated_edges, kernel, iterations=2)
    _save_processed_image_step(closed_edges, "05_closed_edges")

    # Find contours - use RETR_EXTERNAL to get only outermost contours
    contours, hierarchy = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy for visualization
    contours_img = img.copy()
    cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 2)
    _save_processed_image_step(contours_img, "06_all_contours")

    if contours:
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Get image dimensions for ratio calculations
        img_area = img.shape[0] * img.shape[1]

        # Try the largest contours until we find a valid document boundary
        for contour in contours[:5]:  # Check top 5 largest contours
            # Filter contours by area - must be at least 10% of image area
            # This ensures we skip small noise but don't reject valid documents
            min_area = img_area * 0.10
            if cv2.contourArea(contour) < min_area:
                continue

            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)

            # Validate the contour represents a reasonable document shape
            aspect_ratio = w / float(h)
            area_ratio = cv2.contourArea(contour) / (w * h)  # How filled the bounding box is
            contour_area = cv2.contourArea(contour)

            # Document should have reasonable aspect ratio and fill most of its bounding box
            # Also check that contour area is reasonable compared to image area
            if 0.5 <= aspect_ratio <= 3.0 and area_ratio > 0.5:
                # Add small padding (1%) to ensure we don't cut off content
                pad_x = int(w * 0.01)
                pad_y = int(h * 0.01)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(img.shape[1], x + w + pad_x)
                y2 = min(img.shape[0], y + h + pad_y)

                # Only crop if the detected area is significantly smaller than the original
                # This prevents unnecessary cropping when document already fills the frame
                if w * h < img_area * 0.95:
                    cropped = img[y1:y2, x1:x2]
                    _save_processed_image_step(cropped, "07_cropped_document")
                    return cropped

    # Return original image if no valid document contour found
    return img


def preprocess_image(img_bytes: bytes, doc_type) -> np.ndarray:
    # Check if content is PDF and convert to image if needed
    if _is_pdf_content(img_bytes):
        img_bytes = _convert_pdf_to_image(img_bytes)

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Imagen inválida o corrupta")

    # Crop out background from the document photo.
    img = _crop_document(img)

    # Only apply white mask to hide footer data for cédula documents
    # RIF and carnet documents need their full content visible
    if doc_type == "cedula":
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

    _save_processed_image(img)
    return img


# def extract_raw_text(img_bytes: bytes, doc_type: str = "cedula") -> str:
#     processed = preprocess_image(img_bytes, doc_type)
#     ocr_result = reader.readtext(processed, detail=1, paragraph=False)
#     if not ocr_result:
#         raise HTTPException(422, "No se detectó texto en la imagen")
#     return " ".join([line[1] for line in ocr_result])


def encode_image_to_base64(img_bytes: bytes, doc_type: str = "cedula") -> str:
    """
    Encode image bytes to base64 string for sending to vision model.
    Preprocesses image (PDF conversion, cropping, cleaning) before encoding.

    Args:
        img_bytes: Raw image or PDF bytes
        doc_type: Type of document (cedula, rif, carnet)

    Returns:
        Base64 encoded image string
    """
    try:
        # Preprocess image: convert PDF if needed, crop document, apply masks
        processed_img = preprocess_image(img_bytes, doc_type)

        # Convert numpy array back to bytes
        _, buffer = cv2.imencode('.png', processed_img)
        img_bytes_processed = buffer.tobytes()

        # Encode to base64
        base64_string = base64.b64encode(img_bytes_processed).decode("utf-8")

        return base64_string
    except Exception as e:
        raise