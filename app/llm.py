import json
import requests
from fastapi import HTTPException
from .config import settings


def build_prompt_cedula(raw_text: str) -> str:
    return f"""
Eres un asistente experto en normalización y extracción de datos de documentos de identidad a partir de texto OCR, con especialización en cédulas venezolanas.

TEXTO OCR DE ENTRADA:
{raw_text}

INSTRUCCIONES:

1. CLASIFICACIÓN DEL DOCUMENTO:
- Analiza el texto para determinar si corresponde a una Cédula de Identidad venezolana.
- Indicadores de cédula venezolana: "CÉDULA DE IDENTIDAD", "REPÚBLICA BOLIVARIANA DE VENEZUELA", "SAIME", cédula con formato V/E + dígitos, campos típicos (nacionalidad: VENEZOLANO/A).
- Si el texto corresponde a una cédula/ID de OTRO país o no se puede determinar con certeza: establece `cedula_ven: false` pero CONTINÚA con la extracción de todos los campos posibles.
- Solo si el texto es completamente irreconocible, no es un documento de identidad, o está vacío: devuelve todos los campos en null y `cedula_ven: false`.

2. EXTRACCIÓN Y NORMALIZACIÓN (aplica en todos los casos):
- cedula: Formato "V" o "E" + 6 a 8 dígitos si es venezolana. La letra inicial de la cédula debe ser V si la nacionalidad es venezolana, o E si la nacionalidad es extranjera. Limpia espacios, puntos, comas.
- nombre / apellido: Conserva caracteres originales. Elimina ruido OCR (@, #, *, espacios dobles). Normaliza a MAYÚSCULAS. No inventes ni completes.
- fecha_nacimiento / fecha_emision: Formato "DD/MM/AAAA". Si el año tiene 2 dígitos, infiere siglo más probable. Si es ilegible: null.
- fecha_expiracion: Formato "MM/AAAA" o "AAAA". Si no está presente: null.
- estado_civil: Valores permitidos: SOLTERO/A, CASADO/A, DIVORCIADO/A, VIUDO/A. Normaliza a MAYÚSCULAS. De ser en femenino pasar a masculino. Si no aplica o no aparece: null.
- nacionalidad: Extrae literalmente (VENEZOLANO, COLOMBIANO, EXTRANJERO, etc.). Si dice "VENEZOLANA", normaliza a "VENEZOLANO" solo en este campo.
- expiro: Si la fecha de expiracion de la poliza es identica o menor a la fecha actual devuelve true, de lo contrario false.

3. FILTRADO DE RUIDO:
- MUI IMPORTANTE: Ignora EXPLÍCITAMENTE los nombres de directores/funcionarios del SAIME. Estos NUNCA deben ir en `nombre` o `apellido`. Nombres conocidos a descartar: Hendrick Jose Perdomo Colmenares, Gustavo Vizcaino, Dante Rivas, Hugo Cabezas, Pedro Rodriguez, Juan Dugarte O Juan Dugarto y cualquier variación similar que aparezca como firma o cargo institucional.
- No ignores el nombre real de la persona si aparece luego de un nombre de director, pero asegúrate de que el nombre del director no se mezcle con el del titular.
- Tambien debes ignorar la palabra director que usualmente esta antes o despues de estos nombres, para evitar confusiones. A veces se lee como Docto o variantes similares, pero si ves "director" o algo parecido seguido de un nombre conocido, descarta ambos.
- No incluyas títulos (Dr., Ing., Gral.) en nombre/apellido a menos que sean parte del registro civil.

4. FORMATO DE SALIDA (ESTRICTO):
- Devuelve ÚNICAMENTE un objeto JSON válido con esta estructura exacta.
- Si un campo no está presente o es ilegible, asigna null.
- NO uses bloques de código (```json), NO añadas explicaciones, NO incluyas texto antes o después del JSON.

{{
  "cedula_ven": "bool",
  "cedula": "string o null",
  "nombre": "string o null",
  "apellido": "string o null",
  "nacionalidad": "string o null",
  "fecha_nacimiento": "DD/MM/AAAA o null",
  "estado_civil": "string o null",
  "fecha_emision": "DD/MM/AAAA o null",
  "fecha_expiracion": "MM/AAAA o null",
  "expiro": "bool"
}}

EJEMPLOS:

Entrada (cédula venezolana):
OCR: eiGe Oen 4 De VEnerudL4 C € D U L a 0 1d @Nid Ad 29,546.994 ARAUJO FERNANDEZ JOSIELY PAOLA 04/10/2000 SOLTERA VENEZOLANO
Salida: {{"cedula_ven":true,"cedula":"V29546994","nombre":"JOSIELY PAOLA","apellido":"ARAUJO FERNANDEZ","nacionalidad":"VENEZOLANO","fecha_nacimiento":"04/10/2000","estado_civil":"SOLTERO","fecha_emision":null,"fecha_expiracion":null,"expiro":null}}


Entrada (cédula extranjera - ej. colombiana):
OCR: REPUBLICA DE COLOMBIA CEDULA DE CIUDADANIA 1.098.765.432 NOMBRE: MARIA LOPEZ GOMEZ 15/03/1990 COLOMBIANA
Salida: {{"cedula_ven":false,"cedula":"1098765432","nombre":"MARIA","apellido":" LOPEZ GOMEZ","nacionalidad":"COLOMBIANA","fecha_nacimiento":"15/03/1990","estado_civil":null,"fecha_emision":null,"fecha_expiracion":null,"expiro":null}}

Entrada (documento irreconocible):
OCR: factura de servicios publicos numero 445-22
Salida: {{"cedula_ven":false,"cedula":null,"nombre":null,"apellido":null,"nacionalidad":null,"fecha_nacimiento":null,"estado_civil":null,"fecha_emision":null,"fecha_expiracion":null,"expiro":null}}

Caso directivo (nombres de funcionarios del SAIME):
OCR:Ropuolica Dolivariana de Venezuoua STPV PELRANLLDAD E 81.241,.717 152 DuPRAT ROJAS Juan Dugarto Docto GlaDYS OlVIA 01/07/1043 Soltera 1 22/10/2013 04/2017 UAn 1 Kesolil Cn4O CH  2n JULADo EXTRANJERO
Salida: {{"cedula_ven":true,"cedula":"E81241717","nombre":"GLADYS OLIVIA","apellido":"DUPRAT ROJAS","nacionalidad":"EXTRANJERO","fecha_nacimiento":"01/07/1943","estado_civil":"SOLTERO","fecha_emision":"22/10/2013","fecha_expiracion":"04/2017","expiro":true}}
"""

def build_prompt_rif(raw_text: str) -> str:
    return f"""
Eres un asistente experto en normalización y extracción de datos de documentos fiscales a partir de texto OCR, con especialización en RIF (Registro Único de Información Fiscal) de Venezuela.

TEXTO OCR DE ENTRADA:
{raw_text}

INSTRUCCIONES:

1. CLASIFICACIÓN DEL DOCUMENTO:
- Analiza el texto para determinar si corresponde a un RIF venezolano.
- Indicadores de RIF venezolano: "REGISTRO ÚNICO DE INFORMACIÓN FISCAL", "RIF", "SENIAT", formato de RIF con letra inicial (V/E/J/G) + 9 dígitos.
- Si el texto NO corresponde a un RIF o no se puede determinar: establece `rif_valido: false` pero CONTINÚA con la extracción de todos los campos posibles.
- Solo si el texto es completamente irreconocible, no es un documento fiscal, o está vacío: devuelve todos los campos en null y `rif_valido: false`.

2. EXTRACCIÓN Y NORMALIZACIÓN (aplica en todos los casos):
- rif: Formato con letra inicial (V, E, J, G) + 9 dígitos. Limpia espacios, puntos, comas. La letra indica: V=Venezolano, E=Extranjero, J=Jurídica, G=Gubernamental.
- nombre: Conserva caracteres originales. Elimina ruido OCR (@, #, *, espacios dobles). Normaliza a MAYÚSCULAS. No inventes ni completes. Puede ser nombre de persona natural o razón social de empresa.
- direccion: Extrae la dirección completa del domicilio fiscal. Incluye calles, edificios, pisos, apartamentos, sectores, etc. Normaliza a MAYÚSCULAS.
- ciudad: Extrae la ciudad/municipio. Normaliza a MAYÚSCULAS.
- estado: Extrae el estado. Normaliza a MAYÚSCULAS.
- fecha_vencimiento: Formato "DD/MM/AAAA". Busca "FECHA DE VENCIMIENTO" o "VENCIMIENTO". Si es ilegible: null.
- expiro: Si la fecha de vencimiento es idéntica o menor a la fecha actual devuelve true, de lo contrario false. Si no hay fecha: null.

3. FILTRADO DE RUIDO:
- MUY IMPORTANTE: Ignora EXPLÍCITAMENTE el texto de pie de página (footer) como: "La validez de este Comprobante debe verificarse través de la dirección www.seniat.gob.ve", "Sistemas en", "Comprobante Digital RIF", "No requiere sello húmedo", "GERENCIA REGIONAL DE TRIBUTOS INTERNOS", nombres de oficinas del SENIAT.
- Ignora también notas sobre retención de impuestos, agentes de retención, exoneraciones, y cualquier texto institucional que no sea parte de los datos del contribuyente.
- El RIF y el nombre suelen estar cercanos uno del otro. Ejemplo: "V278424924 MANUEL ALEXANDER VALBUENA RODRIGUEZ".
- El domicilio fiscal suele tener este orden: direccion, ciudad, estado. Ejemplo: "CALLE CALLE 73 EDIF HALEAKALA PISO PB APT APTO. 3 SECTOR LA LAGO MARACAIBO ZULIA".

4. FORMATO DE SALIDA (ESTRICTO):
- Devuelve ÚNICAMENTE un objeto JSON válido con esta estructura exacta.
- Si un campo no está presente o es ilegible, asigna null.
- NO uses bloques de código (```json), NO añadas explicaciones, NO incluyas texto antes o después del JSON.

{{
  "rif_valido": "bool",
  "rif": "string o null",
  "nombre": "string o null",
  "estado": "string o null",
  "ciudad": "string o null",
  "direccion": "string o null",
  "expiro": "bool o null"
}}

EJEMPLOS:

Entrada (RIF de persona natural):
OCR: Republica Bolivariana de Venezuela SENIAT REGISTRO ÚNICO DE INFORMACIÓN FISCAL (RIF) V278424924 MANUEL ALEXANDER VALBUENA RODRIGUEZ FECHA DE INSCRIPCION DOMICILIO FISCAL CALLE CALLE 73 EDIF HALEAKALA PISO PB APT APTO. 3 SECTOR LA LAGO MARACAIBO ZULIA ZONA POSTAL 4001 FECHA DE VENCIMIENTO 15/12/2025
Salida: {{"rif_valido":true,"rif":"V278424924","nombre":"MANUEL ALEXANDER VALBUENA RODRIGUEZ","estado":"ZULIA","ciudad":"MARACAIBO","direccion":"CALLE CALLE 73 EDIF HALEAKALA PISO PB APT APTO. 3 SECTOR LA LAGO","expiro":false}}

Entrada (RIF de empresa):
OCR: REPUBLICA BOLIVARIANA DE VENEZUELA SENIAT REGISTRO UNICO DE INFORMACION FISCAL J123456789 EMPRESA EJEMPLO C.A. DOMICILIO FISCAL AV PRINCIPAL EDIF CORPORATIVO PISO 5 CARACAS DISTRITO CAPITAL VENCIMIENTO 30/06/2024
Salida: {{"rif_valido":true,"rif":"J123456789","nombre":"EMPRESA EJEMPLO C.A.","estado":"DISTRITO CAPITAL","ciudad":"CARACAS","direccion":"AV PRINCIPAL EDIF CORPORATIVO PISO 5","expiro":true}}

Entrada (documento irreconocible):
OCR: factura de servicios publicos numero 445-22
Salida: {{"rif_valido":false,"rif":null,"nombre":null,"estado":null,"ciudad":null,"direccion":null,"expiro":null}}
"""

def build_prompt_carnet(raw_text: str) -> str:
    # DEJADO VACÍO COMO SOLICITASTE
    return ""

def get_prompt(doc_type: str, raw_text: str) -> str:
    if doc_type == "cedula":
        return build_prompt_cedula(raw_text)
    elif doc_type == "rif":
        return build_prompt_rif(raw_text)
    elif doc_type == "carnet":
        return build_prompt_carnet(raw_text)
    return ""

def call_llm(raw_text: str, doc_type: str = "cedula") -> str:
    prompt = get_prompt(doc_type, raw_text)

    # 🛡️ Seguridad: Si el prompt está vacío (RIF o Carnet), devolvemos "{}" 
    # para evitar que el LLM falle mientras terminas de configurarlos.
    if not prompt.strip():
        return "{}"

    if not settings.GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY no configurada")

    request_options = {
        "headers": {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        "json": {
            "model": settings.GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1024,
        },
        "timeout": 30,
        "verify": settings.GROQ_VERIFY_SSL,
    }
    if settings.REQUESTS_CA_BUNDLE:
        request_options["verify"] = settings.REQUESTS_CA_BUNDLE

    try:
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", **request_options)
        res.raise_for_status()
        response_data = res.json()
    except requests.exceptions.SSLError as e:
        print("SSL Error:", str(e))
        raise HTTPException(
            502,
            "Error de SSL al contactar Groq. Si estás en un entorno local, revisa tu CA o usa GROQ_VERIFY_SSL=false solo para desarrollo.",
        )
    except requests.exceptions.RequestException as e:
        print("Request Error:", str(e))
        raise HTTPException(500, f"Error en LLM: {str(e)}")

    print("LLM Response:", response_data)
    if not isinstance(response_data, dict):
        raise HTTPException(502, "Respuesta inválida del modelo de IA")

    choices = response_data.get("choices")
    if not choices or not isinstance(choices, list):
        raise HTTPException(502, "Respuesta inválida del modelo de IA")

    first_choice = choices[0]
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise HTTPException(502, "Respuesta inválida del modelo de IA")

    return content.strip()
