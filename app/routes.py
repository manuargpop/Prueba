from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from .ocr import extract_raw_text
from .llm import call_llm
from .schemas import CedulaData, RifData, CarnetData
import json

router = APIRouter(prefix="/api/v1")

async def process_file(file: UploadFile, doc_type: str) -> dict:
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        filename = file.filename or ""
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp")):
            raise HTTPException(400, f"Solo se aceptan imágenes para {doc_type}")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(400, f"Imagen {doc_type} mayor a 10MB")

    raw_text = extract_raw_text(content)
    print(f"Raw OCR Text ({doc_type}):", raw_text)

    llm_output = call_llm(raw_text, doc_type)
    json_str = llm_output.replace("```json", "").replace("```", "").strip()
    
    try:
        data_dict = json.loads(json_str)
    except json.JSONDecodeError:
        raise HTTPException(502, f"Respuesta inválida del modelo de IA para {doc_type}")
        
    return data_dict

@router.post("/extract-documents")
async def extract_documents(
    ceddoc: Optional[UploadFile] = File(None),
    rifnat: Optional[UploadFile] = File(None),
    cardoc: Optional[UploadFile] = File(None)
):
    # Inicializamos la respuesta con las 3 claves en None
    result = {
        "ceddoc": None,
        "rifnat": None,
        "cardoc": None
    }

    if ceddoc:
        data_dict = await process_file(ceddoc, "cedula")
        result["ceddoc"] = CedulaData(**data_dict).model_dump()
        
    if rifnat:
        data_dict = await process_file(rifnat, "rif")
        result["rifnat"] = RifData(**data_dict).model_dump()
        
    if cardoc:
        data_dict = await process_file(cardoc, "carnet")
        result["cardoc"] = CarnetData(**data_dict).model_dump()

    # Devolvemos JSONResponse para garantizar que las claves None se incluyan
    return JSONResponse(content=result)