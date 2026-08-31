"""
fase1b_agrupacion_categorica.py
================================

FASE 1B (comun a ambas ramas, corre ANTES del fork supervisado/no supervisado)
- Reduccion de dimensionalidad de categoricas de cardinalidad muy alta,
agrupando nombres lexicamente similares.

Motivacion
----------
La fase 1 ya cubre dos formas de "demasiada cardinalidad":

* Si una categoria domina la muestra (>=90%, `flg_dominancia_alta`) o casi
  toda (>=99%, elimina), el problema es DESBALANCE, no dispersion.
* La fase 2 (rama supervisada) agrupa por FRECUENCIA en ``__OTROS__``
  (``max_categorias``/``min_prop_bin``), pero eso solo conserva las N
  categorias mas frecuentes y descarta el resto sin mirar si comparten algun
  patron entre si (dos sucursales de la misma cadena regional quedan tan
  separadas como una sucursal y un producto).

Queda sin cubrir el caso intermedio: una categorica genuinamente dispersa
(ninguna categoria domina) pero con MUCHOS niveles (ej. +100 nombres de
sucursal, producto o comercio) donde el one-hot (o incluso el binning directo
de la fase 2) diluye la senal en columnas casi vacias. Esta fase ataca
exactamente ese caso: agrupa los NOMBRES por similitud lexica (TF-IDF de
3-gramas + K-Means, k elegido por silueta -- ver
``metricas.agrupar_categoria_por_similitud_nombre`` para el detalle y las
referencias) antes de que la variable llegue a la fase 2, reduciendo N
niveles a k grupos con nombres reconocibles.

Por que corre ANTES del fork y no dentro de cada rama
------------------------------------------------------
No usa el target (agrupa por como se ESCRIBE el nombre, no por como se
COMPORTA frente al target), asi que no hay razon para calcularlo dos veces
-- ni para que las dos ramas puedan llegar a agrupar la misma variable de
forma distinta. Se ejecuta una sola vez sobre el DataFrame completo, justo
despues de la fase 1, y el resultado (una COPIA del DataFrame con las
columnas candidatas reescritas) es lo que consumen fase 2 (bivariada o no
supervisada), fase 3 y Boruta por igual.
"""

from __future__ import annotations

import pandas as pd

from .config import ConfigPipeline
from .logging_utils import obtener_logger
from .metricas import agrupar_categoria_por_similitud_nombre

LOGGER = obtener_logger("fase1b")


def _columnas_candidatas(uni: pd.DataFrame, cfg: ConfigPipeline, sobrevivientes: list[str]) -> list[str]:
    """Categoricas con cardinalidad muy alta y SIN categoria dominante.

    Ambas condiciones a la vez: si ya domina una categoria (>= 90%, zona de
    aviso de la fase 1), el problema es otro y clusterizar no lo resuelve.
    """
    if uni.empty:
        return []
    candidatas = uni[
        (uni["columna"].isin(sobrevivientes))
        & (uni["tipo_inferido"] == "CATEGORICA")
        & (uni["n_unicos"] > cfg.umbral_cardinalidad_clustering)
        & (uni["pct_valor_mas_frecuente"] < cfg.umbral_dominancia_aviso)
    ]
    return candidatas["columna"].tolist()


def ejecutar(
    df: pd.DataFrame, cfg: ConfigPipeline, tipos: dict[str, str],
    sobrevivientes: list[str], uni: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrupa por similitud de nombre las categoricas que califican.

    Returns
    -------
    (df_transformado, reporte)
        ``df_transformado`` es una COPIA de ``df``: las columnas que
        calificaron quedan con sus valores reescritos a la etiqueta de su
        grupo (o sin tocar, si el clustering se omitio para esa columna).
        Todas las demas columnas quedan identicas. ``reporte`` tiene una fila
        por categoria original agrupada, para trazabilidad.
    """
    LOGGER.info("=" * 78)
    LOGGER.info("FASE 1B - AGRUPACION DE CATEGORICAS DE CARDINALIDAD MUY ALTA POR NOMBRE")
    LOGGER.info("=" * 78)

    if not cfg.usar_agrupacion_categorica_nombre:
        LOGGER.info("Omitida (usar_agrupacion_categorica_nombre=False).")
        return df, pd.DataFrame()

    candidatas = _columnas_candidatas(uni, cfg, sobrevivientes)
    if not candidatas:
        LOGGER.info(
            "Ninguna columna califica (cardinalidad > %d y categoria mas frecuente < %.0f%%).",
            cfg.umbral_cardinalidad_clustering, cfg.umbral_dominancia_aviso * 100,
        )
        return df, pd.DataFrame()

    LOGGER.info(
        "%d columna(s) califican para clustering de nombres: %s",
        len(candidatas), ", ".join(candidatas),
    )

    df_out = df.copy()
    reportes: list[pd.DataFrame] = []

    for col in candidatas:
        serie_agrupada, tabla, metodo = agrupar_categoria_por_similitud_nombre(
            df[col], k_max=cfg.max_k_agrupacion_categorica, semilla=cfg.semilla,
        )
        if tabla.empty:
            LOGGER.warning("   - %s -> %s", col, metodo)
            continue

        df_out[col] = serie_agrupada
        tabla.insert(0, "columna", col)
        reportes.append(tabla)
        n_grupos = tabla["etiqueta_grupo"].nunique()
        LOGGER.info("   - %s: %s", col, metodo)
        LOGGER.info(
            "     %d niveles originales -> %d grupos (reduccion %.0f%%)",
            tabla["n_categorias_originales"].iloc[0], n_grupos,
            100 * (1 - n_grupos / tabla["n_categorias_originales"].iloc[0]),
        )

    reporte = pd.concat(reportes, ignore_index=True) if reportes else pd.DataFrame()
    LOGGER.info(
        "Columnas efectivamente agrupadas: %d de %d candidatas.",
        len(reportes), len(candidatas),
    )
    return df_out, reporte
