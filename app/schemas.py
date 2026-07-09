import re
from typing import Optional
from pydantic import BaseModel, field_validator


class CedulaData(BaseModel):
    cedula: Optional[str] = None
    nombre: Optional[str] = None
    apellido: Optional[str] = None
    nacionalidad: Optional[str] = None
    fecha_nacimiento: Optional[str] = None
    estado_civil: Optional[str] = None
    fecha_emision: Optional[str] = None
    fecha_expiracion: Optional[str] = None
    cedula_ven: Optional[bool] = None
    expiro: Optional[bool] = None

    @field_validator("cedula")
    @classmethod
    def clean_cedula(cls, v):
        if not v:
            return None
        clean = re.sub(r"[^VEve0-9]", "", v).upper()
        return clean if re.match(r"^[VE]\d{7,8}$", clean) else v

class RifData(BaseModel):
    
    rif_valido: Optional[bool] = None
    rif: Optional[str] = None
    nombre: Optional[str] = None
    estado: Optional[str] = None
    ciudad: Optional[str] = None
    direccion: Optional[str] = None
    expiro: Optional[bool] = None

    @field_validator("rif")
    @classmethod
    def clean_rif(cls, v):
        if not v:
            return None
        # Remove all non-alphanumeric characters except V, E, J, G and digits
        clean = re.sub(r"[^VEJGvejg0-9]", "", v).upper()
        # Validate: must start with V, E, J, G followed by exactly 9 digits
        return clean if re.match(r"^[VEJG]\d{9}$", clean) else v

class CarnetData(BaseModel):
    # Placeholder vacío, rellena los campos cuando tengas tu prompt listo
    placa: Optional[str] = None

class ExtractResponse(BaseModel):
    ceddoc: Optional[CedulaData] = None
    rifnat: Optional[RifData] = None
    cardoc: Optional[CarnetData] = None