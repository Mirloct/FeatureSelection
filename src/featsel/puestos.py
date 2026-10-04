"""Preparacion determinista de puestos para el contexto KDE y la exportacion.

Data sources / inputs: despuestocolaborador o desposicioncolaborador del dataset;
opciones de puestos_colaborador en config.yaml.
Created: 2026-10-02
Last modified: 2026-10-02
Changelog:
- 2026-10-02: prioridad de fuente, reparacion de texto, agrupacion y contexto KDE.
"""

from __future__ import annotations

from copy import deepcopy
import re
import unicodedata

import pandas as pd

from .config import ConfigPipeline
from .logging_utils import obtener_logger

LOGGER = obtener_logger("puestos")

# Reglas lexicas, sin aprendizaje sobre periodos futuros ni coincidencias difusas.
_FAMILIAS = {
    "GTE": "GERENTE", "GERENTE": "GERENTE", "GERENTA": "GERENTE",
    "ASIST": "ASISTENTE", "ASISTENTE": "ASISTENTE", "ASISTENTA": "ASISTENTE",
    "ASISTENT": "ASISTENTE", "ASISTETE": "ASISTENTE", "ASITENTE": "ASISTENTE",
    "ASISTRNTE": "ASISTENTE", "AST": "ASISTENTE",
    "ANALISTA": "ANALISTA", "ANALIST": "ANALISTA", "ANAL": "ANALISTA",
    "ANALST": "ANALISTA", "ANL": "ANALISTA",
    "SUBGTE": "SUBGERENTE", "SUBGERENTE": "SUBGERENTE",
    "JEFE": "JEFE", "JEFATURA": "JEFE",
    "SUPERV": "SUPERVISOR", "SUPERVISOR": "SUPERVISOR", "SUPERVISORA": "SUPERVISOR",
    "COORD": "COORDINADOR", "COORDINADOR": "COORDINADOR", "COORDINADORA": "COORDINADOR",
    "AUX": "AUXILIAR", "AUXILIAR": "AUXILIAR",
    "TECNICO": "TECNICO", "TECNICA": "TECNICO",
    "EJEC": "EJECUTIVO", "EJECUTIVO": "EJECUTIVO", "EJECUTIVA": "EJECUTIVO",
    "PRACT": "PRACTICANTE", "PRACTICANTE": "PRACTICANTE",
}
_CALIFICADORES = {
    "ADJ": "ADJUNTO", "ADJUNTO": "ADJUNTO", "ADJUNTA": "ADJUNTO",
    "JR": "JUNIOR", "JUNIOR": "JUNIOR", "SR": "SENIOR", "SENIOR": "SENIOR",
}
_VACIOS = {"NAN", "NONE", "NULL", "NA", "N A", "S N", "SIN DATO", "SIN DATOS",
           "SIN INFORMACION", "NO APLICA"}


def _reparar_encoding(texto: str) -> str:
    """Repara mojibake reversible; conserva el original si la conversion falla."""
    def errores(valor: str) -> int:
        return sum(valor.count(c) for c in ("Ã", "Â", "�", "ƒ", "±", "©", "³"))

    for _ in range(2):
        candidatos = [texto]
        for encoding in ("cp1252", "latin1"):
            try:
                candidatos.append(texto.encode(encoding).decode("utf-8"))
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        mejor = min(candidatos, key=errores)
        if errores(mejor) >= errores(texto):
            break
        texto = mejor
    return texto


def agrupar_puesto(valor: object, sin_dato: str) -> str:
    """Usa hasta dos palabras limpias; para puestos desconocidos usa la primera."""
    if valor is None or valor is pd.NA:
        return sin_dato
    texto = _reparar_encoding(str(valor))
    texto = unicodedata.normalize("NFKD", texto).upper()
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    # El signo de reemplazo dentro de una palabra indica una letra irrecuperable.
    # Se elimina para reconocer, por ejemplo, ANAL�STA y T�CNICO.
    texto = texto.replace("�", "")
    palabras = re.findall(r"[A-Z]+", texto)
    limpio = " ".join(palabras)
    if not palabras or limpio in _VACIOS:
        return sin_dato
    primera, segunda = (palabras + [""])[:2]
    if primera in {"GTEADJ", "GERENTEADJ", "GERENTEADJUNTO", "GTEADJUNTO"}:
        return "GERENTE ADJUNTO"
    if primera in {"TCNICO", "TECNICO"}:
        primera = "TECNICO"
    if primera == "ANALSTA":
        primera = "ANALISTA"
    if primera == "SUB" and segunda in {"GTE", "GERENTE"}:
        return "SUBGERENTE"
    familia = _FAMILIAS.get(primera)
    if familia:
        calificador = _CALIFICADORES.get(segunda)
        return f"{familia} {calificador}" if calificador else familia
    return primera


def preparar(df: pd.DataFrame, cfg: ConfigPipeline) -> tuple[pd.DataFrame, ConfigPipeline]:
    """Elige una fuente, crea el grupo y devuelve una configuracion independiente.

    Sin fuente o con el interruptor apagado, devuelve los objetos sin cambios.
    No activa FE ni inventa conductas; conserva los demas contextos configurados.
    """
    if not cfg.usar_estandarizacion_puesto:
        return df, cfg
    # Se toleran diferencias de mayusculas, espacios y BOM en el encabezado.
    disponibles = {str(c).strip().lstrip("\ufeff").casefold(): c for c in df.columns}
    fuentes = [disponibles[n.casefold()] for n in cfg.columnas_puesto_prioridad
               if n.casefold() in disponibles]
    if not fuentes:
        LOGGER.info("Puestos: no hay columna de origen; se omite la estandarizacion.")
        return df, cfg
    fuente = fuentes[0]
    salida = cfg.columna_puesto_agrupado
    resultado = df.copy()
    # Normaliza valores distintos una vez: no reprocesa el mismo puesto por fila.
    texto = resultado[fuente].astype("string")
    mapa = {valor: agrupar_puesto(valor, cfg.puesto_sin_dato) for valor in texto.dropna().unique()}
    resultado[salida] = texto.map(mapa).fillna(cfg.puesto_sin_dato).astype("string")
    sobrantes = fuentes[1:]
    resultado = resultado.drop(columns=sobrantes)
    efectivo = deepcopy(cfg)
    efectivo.columnas_conservadas = list(dict.fromkeys([
        *(c for c in cfg.columnas_conservadas if c not in sobrantes), fuente, salida,
    ]))
    # Reemplaza el contexto original por el grupo; otros contextos siguen vigentes.
    nombres_fuente = {n.casefold() for n in cfg.columnas_puesto_prioridad}
    efectivo.context_vars = list(dict.fromkeys([
        *(c for c in cfg.context_vars
          if c.casefold() not in nombres_fuente and c != salida
          and not (c == "job_position" and c not in resultado.columns)), salida,
    ]))
    LOGGER.info("Puestos: fuente='%s', agrupado='%s', %d grupos; se retiraron %s.",
                fuente, salida, resultado[salida].nunique(), sobrantes or "ninguna columna")
    if resultado[salida].eq(cfg.puesto_sin_dato).any():
        LOGGER.warning("Puestos sin texto util: %d filas usan '%s'.",
                       int(resultado[salida].eq(cfg.puesto_sin_dato).sum()), cfg.puesto_sin_dato)
    return resultado, efectivo
