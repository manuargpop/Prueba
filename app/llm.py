import base64
import requests
from fastapi import HTTPException
from datetime import datetime
from openai import OpenAI
from .config import settings


def build_prompt_cedula() -> str:
    current_date = datetime.now().strftime("%d/%m/%Y")
    return f"""
Eres un asistente experto en análisis y extracción de datos de documentos de identidad a partir de imágenes, con especialización en cédulas venezolanas.

ANALIZA LA IMAGEN ADJUNTA y extrae la información siguiendo estas instrucciones:

1. CLASIFICACIÓN DEL DOCUMENTO:
- Determina si la imagen corresponde a una Cédula de Identidad venezolana.
- Indicadores de cédula venezolana: "CÉDULA DE IDENTIDAD", "REPÚBLICA BOLIVARIANA DE VENEZUELA", "SAIME", cédula con formato V/E + dígitos, campos típicos (nacionalidad: VENEZOLANO/A).
- Si la imagen corresponde a una cédula/ID de OTRO país o no se puede determinar con certeza: establece `cedula_ven: false` pero CONTINÚA con la extracción de todos los campos posibles.
- Solo si la imagen es completamente irreconocible, no es un documento de identidad, o está vacía: devuelve todos los campos en null y `cedula_ven: false`.

2. EXTRACCIÓN Y NORMALIZACIÓN (aplica en todos los casos):
- cedula: Formato "V" o "E" + 6 a 8 dígitos si es venezolana. La letra inicial de la cédula debe ser V si la nacionalidad es venezolana, o E si la nacionalidad es extranjera. Limpia espacios, puntos, comas.
- nombre / apellido: Conserva caracteres originales. Elimina ruido OCR (@, #, *, espacios dobles). Normaliza a MAYÚSCULAS. No inventes ni completes.
- fecha_nacimiento / fecha_emision: Formato "DD/MM/AAAA". Si el año tiene 2 dígitos, infiere siglo más probable. Si es ilegible: null.
- fecha_expiracion: Formato "MM/AAAA" o "AAAA". Si no está presente: null.
- estado_civil: Valores permitidos: SOLTERO/A, CASADO/A, DIVORCIADO/A, VIUDO/A. Normaliza a MAYÚSCULAS. De ser en femenino pasar a masculino. Si no aplica o no aparece: null.
- sexo: Extrae el género basado en el estado civil. Si el estado civil termina en "A" (SOLTERA, CASADA, DIVORCIADA, VIUDA) devuelve "F". Si termina en "O" (SOLTERO, CASADO, DIVORCIADO, VIUDO) devuelve "M". Si estado_civil es null o no se puede determinar: null.
- nacionalidad: Extrae literalmente (VENEZOLANO, COLOMBIANO, EXTRANJERO, etc.). Si dice "VENEZOLANA", normaliza a "VENEZOLANO" solo en este campo.
- expiro: Compara la fecha_expiracion con la fecha actual ({current_date}). Si la fecha de expiración es idéntica o menor a la fecha actual devuelve true, de lo contrario false.

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
  "sexo": "M o F o null",
  "fecha_emision": "DD/MM/AAAA o null",
  "fecha_expiracion": "MM/AAAA o null",
  "expiro": "bool"
}}
"""


def build_prompt_rif() -> str:
    current_date = datetime.now().strftime("%d/%m/%Y")
    return ""
"""
Eres un asistente experto en análisis y extracción de datos de documentos fiscales a partir de imágenes, con especialización en RIF (Registro Único de Información Fiscal) de Venezuela.

ANALIZA LA IMAGEN ADJUNTA y extrae la información siguiendo estas instrucciones:

1. CLASIFICACIÓN DEL DOCUMENTO:
- Determina si la imagen corresponde a un RIF venezolano.
- Indicadores de RIF venezolano: "REGISTRO ÚNICO DE INFORMACIÓN FISCAL", "RIF", "SENIAT", formato de RIF con letra inicial (V/E/J/G) + 9 dígitos.
- Si la imagen NO corresponde a un RIF o no se puede determinar: establece `rif_valido: false` pero CONTINÚA con la extracción de todos los campos posibles.
- Solo si la imagen es completamente irreconocible, no es un documento fiscal, o está vacía: devuelve todos los campos en null y `rif_valido: false`.

2. EXTRACCIÓN Y NORMALIZACIÓN (aplica en todos los casos):
- rif: Formato con letra inicial (V, E, J, G) + 9 dígitos. Limpia espacios, puntos, comas. La letra indica: V=Venezolano, E=Extranjero, J=Jurídica, G=Gubernamental.
- nombre: Conserva caracteres originales. Elimina ruido (@, #, *, espacios dobles). Normaliza a MAYÚSCULAS. No inventes ni completes. Puede ser nombre de persona natural o razón social de empresa.
- direccion: Extrae la dirección completa del domicilio fiscal. Incluye calles, edificios, pisos, apartamentos, sectores, etc. Normaliza a MAYÚSCULAS.
- ciudad: Extrae la ciudad/municipio. Normaliza a MAYÚSCULAS.
- estado: Extrae el estado. Normaliza a MAYÚSCULAS.
- fecha_vencimiento: Formato "DD/MM/AAAA". Busca "FECHA DE VENCIMIENTO" o "VENCIMIENTO". Si es ilegible: null.
- expiro: Compara la fecha_vencimiento con la fecha actual ({current_date}). Si la fecha de vencimiento es idéntica o menor a la fecha actual devuelve true, de lo contrario false. Si no hay fecha: null.

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
"""


def build_prompt_carnet() -> str:
    return ""
"""
Eres un asistente experto en análisis y extracción de datos de documentos vehiculares a partir de imágenes, con especialización en el Certificado de Registro de Vehículo (Carnet de Circulación) de Venezuela emitido por el INTT.

ANALIZA LA IMAGEN ADJUNTA y extrae la información siguiendo estas instrucciones:

1. CLASIFICACIÓN DEL DOCUMENTO:
- Determina si la imagen corresponde a un Carnet de Circulación / Certificado de Registro de Vehículo venezolano.
- Indicadores de carnet válido: "INTT", "INSTITUTO NACIONAL DE TRANSPORTE TERRESTRE", "CERTIFICADO DE REGISTRO DE VEHÍCULO", "CERTIFICADO DE CIRCULACIÓN", campos típicos como Placa, Serial, Carrozado, Motor, Modelo, Marca, Año, Color, Nro. Puestos/PTOS.
- Si la imagen NO corresponde a un carnet vehicular o no se puede determinar: establece `carnet_valido: false` pero CONTINÚA con la extracción de todos los campos posibles.
- Solo si la imagen es completamente irreconocible, no es un documento vehicular, o está vacía: devuelve todos los campos en null y `carnet_valido: false`.

2. EXTRACCIÓN Y NORMALIZACIÓN (aplica en todos los casos):
- placa: Formato típico venezolano: 2-3 letras seguidas de 4-6 dígitos, o combinaciones alfanuméricas (ej: AD313UV, ABC123, A123BC). Limpia espacios, puntos, comas. Normaliza a MAYÚSCULAS. No inventes caracteres.
- cedula_rif: Formato con letra inicial (V, E, J, G) seguida de 7-9 dígitos. Puede aparecer como "Cédula", "RIF", "CÉDULA O RIF". Limpia espacios, puntos, comas. La letra indica: V=Venezolano, E=Extranjero, J=Jurídica, G=Gubernamental.
- pasajeros: Número entero que indica la capacidad de puestos del vehículo. Busca cerca de "Nro. Puestos", "PTOS", "PUESTOS", "CAPACIDAD". Si aparece como "5 PTOS" o "Nro. Puestos: 5", extrae solo el número (5). Si es ilegible o no aparece: null.
- expiro: Este campo NO APLICA para carnets de circulación. Siempre devuelve null.

3. FILTRADO DE RUIDO:
- MUY IMPORTANTE: Ignora EXPLÍCITAMENTE el texto institucional repetitivo como: "INTT", "INSTITUTO NACIONAL DE TRANSPORTE TERRESTRE", "Gobierno Bolivariano de Venezuela", "Ministerio del Poder Popular para el Transporte", "SEREFICAP", "Certificado de Circulación Para ser archivado en lugar seguro", "Necesario para salir del Territorio Nacional", "Este documento acredita propiedad sobre el vehículo", "Debe ser impreso color y solo en papel BOND blanco", "Se recomienda plastificar este carnet", números de autorización largos, códigos de verificación, hashes, timestamps repetidos.
- El mismo valor (placa, cédula/rif, serial) puede aparecer 2-3 veces en diferentes secciones del documento. Usa la primera ocurrencia clara y consistente. Si hay discrepancias, prioriza la versión que aparece en la sección principal de datos del vehículo.
- Ignora textos como "TC: GAS", "Carrozado", "Serial Chasis", "Serial Motor", "NIV", "Color", "Marca", "Modelo", "Año", "Clase", "Tipo", "Uso", "Servicio", "Carga", "Tara", "Ejes" a menos que sean necesarios para confirmar que es un carnet válido. Solo extrae placa, cedula_rif y pasajeros.
- Los valores de cedula_rif y placa suelen estar cerca del nombre del propietario. Ejemplo: "YOLEMAR TERESA GALLARDO PEROZO Cédula O RIF: V19215064 Placa: AD313UV".

4. FORMATO DE SALIDA (ESTRICTO):
- Devuelve ÚNICAMENTE un objeto JSON válido con esta estructura exacta.
- Si un campo no está presente o es ilegible, asigna null.
- NO uses bloques de código (```json), NO añadas explicaciones, NO incluyas texto antes o después del JSON.

{{
  "carnet_valido": "bool",
  "placa": "string o null",
  "cedula_rif": "string o null",
  "pasajeros": "int o null"
}}
"""


def get_prompt(doc_type: str) -> str:
    if doc_type == "cedula":
        return build_prompt_cedula()
    elif doc_type == "rif":
        return build_prompt_rif()
    elif doc_type == "carnet":
        return build_prompt_carnet()
    return ""


def call_llm_with_image(image_base64: str, doc_type: str = "cedula") -> str:
    """
    Call Qwen model via Alibaba Model Studio using OpenAI-compatible API.

    Args:
        image_base64: Base64 encoded image string
        doc_type: Type of document (cedula, rif, carnet)

    Returns:
        JSON string response from the model
    """
    prompt = get_prompt(doc_type)

    if not prompt.strip():
        return "{}"

    if not settings.ALIBABA_API_KEY:
        raise HTTPException(500, "ALIBABA_API_KEY no configurada")

    try:
        print("Enviando request a alibaba")
        # Initialize OpenAI client with Alibaba's compatible endpoint
        client = OpenAI(
            api_key=settings.ALIBABA_API_KEY,
            base_url=settings.ALIBABA_BASE_URL,
        )

        # Prepare the message with image in OpenAI-compatible format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }
        ]

        # Call the model
        completion = client.chat.completions.create(
            model=settings.ALIBABA_MODEL,
            messages=messages,
            temperature=0.6,
            max_tokens=4096,
            top_p=0.95,
        )

        # Extract the response content
        content = completion.choices[0].message.content

        # 🔍 DEBUG: Print raw response from Alibaba Qwen
        print("\n" + "="*50)
        print("🤖 RAW LLM RESPONSE (Alibaba Qwen):")
        print("="*50)
        if content:
            print(content)
        else:
            print("⚠️ No content in response!")
        print("="*50 + "\n")

        if not isinstance(content, str):
            raise HTTPException(502, "Respuesta inválida del modelo de IA")

        return content.strip()

    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            raise HTTPException(500, "Error de autenticación con Alibaba Model Studio. Verifica tu API key.")
        elif "model" in error_msg.lower():
            raise HTTPException(500, f"Error con el modelo {settings.ALIBABA_MODEL}. Verifica que tengas acceso a este modelo.")
        else:
            raise HTTPException(500, f"Error en LLM: {error_msg}")