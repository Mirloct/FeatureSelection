#!/usr/bin/env python
"""
generar_datos_demo.py
=====================

Generador de un PANEL SINTETICO de demostracion.

Sirve para dos cosas:

1. Permitir ejecutar el pipeline end-to-end sin datos reales.
2. Actuar como banco de pruebas: el panel incluye, a proposito, variables con
   cada una de las patologias que el pipeline debe detectar, de modo que se
   puede verificar que cada criterio hace lo que promete.

Patologias sembradas deliberadamente
------------------------------------
====================================  ==================================================
Variable                              Debe ser detectada por
====================================  ==================================================
``var_constante``                     Fase 1.2 - varianza cero
``var_casi_constante``                Fase 1.2 - dominancia de una categoria
``var_cv_bajo``                       Fase 1.2 - coeficiente de variacion despreciable
``var_muchos_ceros``                  Fase 1.1 - exceso de ceros
``var_muchos_nulos``                  Fase 1.1 - exceso de nulos
``var_ruido_1..5``                    Fase 2 - IV y Gini por debajo del piso
``var_clon_score``                    Fase 3 - redundancia con ``var_score_riesgo``
``var_combinacion_lineal``            Fase 3 - VIF alto (colinealidad multiple)
``cat_alta_cardinalidad``             Fase 2 - agrupacion en ``__OTROS__``
``var_fuga``                          Fase 2 - IV sospechosamente alto (leakage)
``var_no_monotona``                   Fase 2 - Gini crudo bajo pero IV alto
``var_inestable_tiempo``              Fase 2 - PSI alto entre periodos
``var_fija_entidad``                  Fase 0 - ICC ~ 1 (invariante en el tiempo)
====================================  ==================================================

Los NOMBRES de las columnas target / id / tiempo se toman del objeto de
configuracion: si se cambian en `config.yaml`, este generador los respeta sin
tocar una linea de codigo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from featsel.config import ConfigPipeline  # noqa: E402
from featsel.logging_utils import obtener_logger  # noqa: E402

LOGGER = obtener_logger("demo")


def generar_panel_demo(
    cfg: ConfigPipeline,
    n_entidades: int = 400,
    n_periodos: int = 18,
    tasa_evento_base: float = 0.12,
) -> Path:
    """Genera y guarda el panel sintetico en ``cfg.ruta_dataset``.

    Parameters
    ----------
    cfg
        Configuracion del pipeline. De aqui salen los nombres de las columnas
        de rol y la ruta de salida, de modo que el demo siempre es coherente
        con lo que el pipeline espera leer.
    n_entidades, n_periodos
        Dimensiones del panel. Por defecto 400 x 18 = 7.200 observaciones.
    tasa_evento_base
        Prevalencia aproximada del evento (target = 1).
    """
    rng = np.random.default_rng(cfg.semilla)

    # --- Esqueleto del panel: producto cartesiano entidad x periodo -------
    entidades = np.array([f"ENT_{i:04d}" for i in range(1, n_entidades + 1)])
    periodos = pd.period_range("2023-01", periods=n_periodos, freq="M").astype(str)

    idx = pd.MultiIndex.from_product([entidades, periodos],
                                     names=[cfg.columna_id, cfg.columna_tiempo])
    df = idx.to_frame(index=False)
    n = len(df)
    t = df.groupby(cfg.columna_id, observed=True).cumcount().to_numpy()  # 0..T-1
    ent_idx = pd.factorize(df[cfg.columna_id])[0]

    # ------------------------------------------------------------------
    # Efectos latentes: heterogeneidad por entidad + tendencia temporal.
    # Es lo que distingue un panel de un corte transversal.
    # ------------------------------------------------------------------
    efecto_entidad = rng.normal(0, 1.0, n_entidades)[ent_idx]
    tendencia = 0.03 * t

    # ------------------------------------------------------------------
    # 1. Variables genuinamente predictivas
    # ------------------------------------------------------------------
    df["var_score_riesgo"] = 550 + 60 * efecto_entidad + rng.normal(0, 25, n) - 4 * t
    df["var_ratio_endeudamiento"] = np.clip(
        0.35 + 0.18 * efecto_entidad + 0.01 * t + rng.normal(0, 0.07, n), 0, 3
    )
    df["var_meses_antiguedad"] = np.clip(24 + 12 * rng.normal(0, 1, n_entidades)[ent_idx] + t, 0, None)
    df["var_utilizacion_linea"] = np.clip(
        0.45 + 0.20 * efecto_entidad + rng.normal(0, 0.12, n), 0, 1.5
    )
    df["var_n_atrasos_12m"] = rng.poisson(np.clip(1.2 + 0.9 * efecto_entidad, 0.05, None))
    df["var_monto_deuda"] = np.exp(9.5 + 0.55 * efecto_entidad + rng.normal(0, 0.45, n))
    df["var_ingreso_estimado"] = np.exp(12.8 - 0.25 * efecto_entidad + rng.normal(0, 0.30, n))

    # Relacion NO monotona: el riesgo sube en los extremos (forma de U).
    # El Gini crudo la subestima; el IV sobre bins la detecta.
    base_u = rng.normal(0, 1, n)
    df["var_no_monotona"] = base_u

    # Variable practicamente fija por entidad (ICC ~ 1).
    df["var_fija_entidad"] = rng.normal(100, 15, n_entidades)[ent_idx]

    # Variable cuya distribucion se desplaza en el tiempo (PSI alto).
    df["var_inestable_tiempo"] = rng.normal(0, 1, n) + 0.45 * t

    # ------------------------------------------------------------------
    # 2. Categoricas
    # ------------------------------------------------------------------
    sectores = ["COMERCIO", "SERVICIOS", "INDUSTRIA", "AGRO", "CONSTRUCCION"]
    riesgo_sector = dict(zip(sectores, [0.0, -0.25, 0.15, 0.40, 0.55]))
    df["cat_sector"] = rng.choice(sectores, n_entidades, p=[.30, .28, .20, .12, .10])[ent_idx]

    df["cat_region"] = rng.choice(["NORTE", "CENTRO", "SUR"], n, p=[.3, .45, .25])
    df["cat_segmento"] = rng.choice(["MASIVO", "PREFERENTE", "PREMIUM"], n_entidades,
                                    p=[.6, .3, .1])[ent_idx]
    # Alta cardinalidad: debe agruparse en __OTROS__ en la fase 2.
    df["cat_alta_cardinalidad"] = rng.choice([f"SUC_{i:03d}" for i in range(180)], n)

    # ------------------------------------------------------------------
    # 3. Target: proceso logistico con efectos de panel
    # ------------------------------------------------------------------
    logit = (
        -2.0
        - 0.0060 * (df["var_score_riesgo"] - 550)
        + 1.80 * (df["var_ratio_endeudamiento"] - 0.35)
        + 0.28 * df["var_n_atrasos_12m"]
        + 1.10 * (df["var_utilizacion_linea"] - 0.45)
        - 0.0040 * df["var_meses_antiguedad"]
        + 0.90 * (base_u**2 - 1.0)                       # efecto en U
        + df["cat_sector"].map(riesgo_sector).to_numpy()
        + tendencia
        + 0.35 * rng.normal(0, 1, n)
    )
    p = 1 / (1 + np.exp(-logit))
    p = np.clip(p * (tasa_evento_base / p.mean()), 0.001, 0.999)
    df[cfg.columna_target] = rng.binomial(1, p)

    # ------------------------------------------------------------------
    # 4. Patologias sembradas
    # ------------------------------------------------------------------
    # 4.1 Constante pura -> fase 1.2
    df["var_constante"] = 7.0
    # 4.2 Casi constante -> dominancia de una categoria
    df["var_casi_constante"] = np.where(rng.random(n) < 0.996, 1, 0)
    # 4.3 Coeficiente de variacion despreciable
    df["var_cv_bajo"] = 1_000_000 + rng.normal(0, 0.02, n)
    # 4.4 Exceso de ceros (~97%)
    df["var_muchos_ceros"] = np.where(rng.random(n) < 0.03, rng.exponential(500, n), 0.0)
    # 4.5 Exceso de nulos (~96%)
    df["var_muchos_nulos"] = np.where(rng.random(n) < 0.04, rng.normal(50, 10, n), np.nan)
    # 4.6 Ruido puro -> fase 2
    for i in range(1, 6):
        df[f"var_ruido_{i}"] = rng.normal(0, 1, n)
    # 4.7 Clon casi perfecto -> fase 3 (redundancia)
    df["var_clon_score"] = df["var_score_riesgo"] * 1.02 + rng.normal(0, 1.2, n)
    # 4.8 Combinacion lineal de otras dos -> VIF alto
    df["var_combinacion_lineal"] = (
        0.6 * df["var_ratio_endeudamiento"] + 0.4 * df["var_utilizacion_linea"]
        + rng.normal(0, 0.01, n)
    )
    # 4.9 Fuga de informacion: construida a partir del propio target
    df["var_fuga"] = df[cfg.columna_target] * 0.85 + rng.normal(0, 0.18, n)
    # 4.10 Nulos dispersos en una variable legitima (realismo)
    mascara_nulos = rng.random(n) < 0.08
    df.loc[mascara_nulos, "var_ingreso_estimado"] = np.nan
    # 4.11 Columna de texto con numeros guardados como string
    df["var_texto_numerico"] = (df["var_monto_deuda"] / 1000).round(2).astype(str)

    # --- Panel deliberadamente desbalanceado: se eliminan observaciones ---
    # Refleja el caso real (entidades que entran y salen del panel) y ejercita
    # la deteccion de desbalance de las validaciones.
    a_eliminar = rng.random(n) < 0.04
    df = df.loc[~a_eliminar].reset_index(drop=True)

    # --- Orden de columnas: roles primero, para lectura humana -------------
    orden = [cfg.columna_id, cfg.columna_tiempo, cfg.columna_target]
    df = df[orden + [c for c in df.columns if c not in orden]]

    ruta = Path(cfg.ruta_dataset)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if ruta.suffix.lower() == ".parquet":
        df.to_parquet(ruta, index=False)
    else:
        df.to_csv(ruta, index=False, sep=cfg.csv_sep, encoding=cfg.csv_encoding)

    LOGGER.info(
        "Panel sintetico generado: %s | %d filas x %d columnas | %d entidades x %d periodos | "
        "tasa de evento=%.4f",
        ruta, df.shape[0], df.shape[1], df[cfg.columna_id].nunique(),
        df[cfg.columna_tiempo].nunique(), df[cfg.columna_target].mean(),
    )
    return ruta


if __name__ == "__main__":
    from featsel.config import cargar_config
    from featsel.logging_utils import configurar_logging

    configurar_logging("outputs/featsel.log")
    generar_panel_demo(cargar_config("config.yaml"))
