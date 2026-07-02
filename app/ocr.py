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


def preprocess_image(img_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Imagen inválida o corrupta")
    	
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
