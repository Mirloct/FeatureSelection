"""
feature_engineering_anomalias.py
================================

Genera variables temporales y de rareza por KDE condicional antes de la
depuracion/seleccion. Tambien prepara una matriz numerica con corte temporal,
apta para Isolation Forest y VAE, sin aprender transformaciones en el holdout.

Data sources / inputs: DataFrame cargado desde ``ConfigPipeline.ruta_dataset``;
columnas id/tiempo, ``context_vars`` y ``behavior_vars`` de ``config.yaml``.
Outputs: DataFrame enriquecido, catalogo/estabilidad de features y CSV numerico
en ``ConfigPipeline.ruta_matriz_anomalias_efectiva``.
Created: 2026-10-01
Last modified: 2026-10-04
Changelog:
- 2026-10-01: implementacion inicial time-safe de lags, ventanas personales,
  KDE conjunta/marginal, catalogo, estabilidad, ablacion y matriz IF/VAE.
- 2026-10-04: protege columnas originales, separa contextos nulos y valida
  infinitos y colisiones de nombres antes de calcular KDE.

La regla central de causalidad es estricta: una fila del mes t solo utiliza
observaciones de meses anteriores. El mes corriente nunca forma parte de su
referencia KDE ni de sus estadisticos personales.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from .config import ConfigPipeline
from .logging_utils import obtener_logger

LOGGER = obtener_logger("feature_engineering_anomalias")
PREFIJO_FE = "fe__"


class ErrorFeatureEngineering(ValueError):
    """Configuracion o datos incompatibles con el feature engineering."""


@dataclass
class ResultadoFeatureEngineering:
    """Artefactos producidos por la etapa previa a la seleccion."""

    dataframe: pd.DataFrame
    catalogo: pd.DataFrame
    estabilidad: pd.DataFrame


@dataclass
class ResultadoPreparacionModelos:
    """Matriz numerica y trazabilidad de transformaciones para IF/VAE."""

    matriz: pd.DataFrame
    esquema: pd.DataFrame
    ruta: str | None


def _validar_columnas(df: pd.DataFrame, cfg: ConfigPipeline) -> None:
    if not df.columns.is_unique:
        raise ErrorFeatureEngineering("El dataset contiene nombres de columnas duplicados.")
    requeridas = [cfg.columna_id, cfg.columna_tiempo, *cfg.context_vars, *cfg.behavior_vars]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ErrorFeatureEngineering(
            "No se puede ejecutar feature engineering; faltan columnas configuradas: "
            f"{faltantes}. Desactive usar_feature_engineering o corrija context_vars/behavior_vars."
        )
    no_numericas = [c for c in cfg.behavior_vars if not pd.api.types.is_numeric_dtype(df[c])]
    if no_numericas:
        raise ErrorFeatureEngineering(
            f"Las behavior_vars deben ser numericas despues de tipificar: {no_numericas}."
        )
    infinitas = [c for c in cfg.behavior_vars
                 if np.isinf(df[c].to_numpy(dtype=float, na_value=np.nan)).any()]
    if infinitas:
        raise ErrorFeatureEngineering(f"Las behavior_vars deben contener valores finitos o nulos: {infinitas}.")
    nombres = []
    for behavior in cfg.behavior_vars:
        nombres.extend(_nombre("temporal", behavior, suffix) for suffix in
                       ["lag1", "diff1", *[f"{stat}_{w}" for w in sorted(set(int(w) for w in cfg.temporal_windows))
                                          for stat in ("mean", "std", "personal_z")]])
        nombres.extend(_nombre("kde", behavior, scope, suffix)
                       for scope, _ in _claves_referencia(cfg)
                       for suffix in ("neglog_density", "tail_rarity"))
    if len(nombres) != len(set(nombres)):
        raise ErrorFeatureEngineering("Hay colisiones en los nombres normalizados de features; renombre las columnas configuradas.")
    colisiones = [c for c in df.columns if str(c).startswith(PREFIJO_FE)]
    if colisiones:
        raise ErrorFeatureEngineering(
            f"El dataset ya contiene columnas con el prefijo reservado '{PREFIJO_FE}': {colisiones[:10]}."
        )


def _periodos_ordenados(serie: pd.Series) -> list[Any]:
    """Orden temporal determinista; falla si los valores no son comparables."""
    if serie.isna().any():
        raise ErrorFeatureEngineering("La columna de tiempo contiene nulos; no es posible evitar look-ahead.")
    try:
        return list(pd.Index(serie.drop_duplicates()).sort_values())
    except TypeError as exc:
        raise ErrorFeatureEngineering(
            "La columna de tiempo mezcla tipos no ordenables. Normalice los meses antes de ejecutar FE."
        ) from exc


def _nombre(*partes: Any) -> str:
    limpio = [str(p).strip().lower().replace(" ", "_").replace("/", "_") for p in partes]
    return PREFIJO_FE + "__".join(limpio)


def _agregar_temporales(
    df: pd.DataFrame, cfg: ConfigPipeline, catalogo: list[dict[str, Any]],
) -> pd.DataFrame:
    """Crea lags y estadisticos personales desplazados un periodo."""
    work = df.reset_index(drop=True).sort_values(
        [cfg.columna_id, cfg.columna_tiempo], kind="mergesort"
    )

    for behavior in cfg.behavior_vars:
        valores = pd.to_numeric(work[behavior], errors="coerce")
        entidad = work[cfg.columna_id]
        historico = valores.groupby(entidad, sort=False).shift(1)
        n_hist = historico.notna().groupby(entidad, sort=False).cumsum()

        lag = _nombre("temporal", behavior, "lag1")
        diff = _nombre("temporal", behavior, "diff1")
        work[lag] = historico
        work[diff] = valores - historico
        catalogo.extend([
            _fila_catalogo(lag, "TEMPORAL", behavior, "personal", "Valor de la entidad en t-1."),
            _fila_catalogo(diff, "TEMPORAL", behavior, "personal", "Cambio actual menos t-1."),
        ])

        for ventana in sorted(set(int(w) for w in cfg.temporal_windows)):
            roll = historico.groupby(entidad, sort=False).rolling(
                # Una ventana corta (p. ej. 3) sigue siendo valida cuando la
                # entidad ya acumulo la historia global minima (p. ej. 6).
                window=ventana, min_periods=min(ventana, cfg.min_personal_history)
            )
            media = roll.mean().reset_index(level=0, drop=True)
            std = roll.std(ddof=0).reset_index(level=0, drop=True)
            media = media.where(n_hist >= cfg.min_personal_history)
            std = std.where(n_hist >= cfg.min_personal_history)

            col_media = _nombre("temporal", behavior, f"mean_{ventana}")
            col_std = _nombre("temporal", behavior, f"std_{ventana}")
            col_z = _nombre("temporal", behavior, f"personal_z_{ventana}")
            work[col_media] = media
            work[col_std] = std
            work[col_z] = (valores - media) / std.clip(lower=cfg.epsilon)
            catalogo.extend([
                _fila_catalogo(col_media, "TEMPORAL", behavior, "personal", f"Media de t-{ventana}..t-1."),
                _fila_catalogo(col_std, "TEMPORAL", behavior, "personal", f"Std de t-{ventana}..t-1."),
                _fila_catalogo(col_z, "TEMPORAL", behavior, "personal", f"Z robusto causal contra t-{ventana}..t-1."),
            ])

    return work.sort_index().reset_index(drop=True)


def _fila_catalogo(
    columna: str, familia: str, behavior: str, referencia: str, definicion: str,
) -> dict[str, Any]:
    return {
        "columna": columna,
        "familia": familia,
        "behavior_var": behavior,
        "referencia": referencia,
        "definicion": definicion,
        "time_safe": 1,
    }


def _escalado_robusto(x: np.ndarray, epsilon: float) -> tuple[np.ndarray, float, float]:
    mediana = float(np.nanmedian(x))
    q25, q75 = np.nanpercentile(x, [25, 75])
    escala = float(q75 - q25)
    if not np.isfinite(escala) or escala <= epsilon:
        escala = float(np.nanstd(x))
    if not np.isfinite(escala) or escala <= epsilon:
        escala = 1.0
    return (x - mediana) / escala, mediana, escala


def _silverman(z: np.ndarray, epsilon: float) -> float:
    n = max(len(z), 2)
    std = float(np.std(z, ddof=1)) if n > 2 else 1.0
    iqr = float(np.subtract(*np.percentile(z, [75, 25])))
    robusta = min(std, iqr / 1.34) if iqr > epsilon else std
    return max(0.9 * max(robusta, epsilon) * n ** (-0.2), 0.05)


def _elegir_bandwidth(
    z: np.ndarray,
    periodos_ref: np.ndarray,
    cfg: ConfigPipeline,
) -> float:
    metodo = cfg.bandwidth_method
    if not isinstance(metodo, str) or metodo not in ("time_safe_cv", "silverman"):
        return float(metodo)
    if metodo == "silverman":
        return _silverman(z, cfg.epsilon)

    unicos = list(pd.Index(periodos_ref).drop_duplicates().sort_values())
    if len(unicos) < 2:
        return _silverman(z, cfg.epsilon)
    ultimo = unicos[-1]
    train = z[periodos_ref != ultimo]
    valid = z[periodos_ref == ultimo]
    if len(train) < max(10, cfg.min_group_size // 2) or len(valid) < 2:
        return _silverman(z, cfg.epsilon)

    mejor_bw = float(cfg.bandwidth_grid[0])
    mejor_score = -np.inf
    for bw in cfg.bandwidth_grid:
        try:
            kde = KernelDensity(kernel=cfg.kernel_type, bandwidth=float(bw)).fit(train[:, None])
            score = float(kde.score(valid[:, None]))
        except ValueError:
            continue
        if score > mejor_score:
            mejor_score, mejor_bw = score, float(bw)
    return mejor_bw


def _claves_referencia(cfg: ConfigPipeline) -> list[tuple[str, list[str]]]:
    claves: list[tuple[str, list[str]]] = []
    if cfg.reference_mode in ("joint", "joint_and_marginal"):
        claves.append(("joint", list(cfg.context_vars)))
    if cfg.reference_mode in ("marginal", "joint_and_marginal"):
        claves.extend((f"marginal_{c}", [c]) for c in cfg.context_vars)
    return claves


def _agregar_kde(
    df: pd.DataFrame, cfg: ConfigPipeline, catalogo: list[dict[str, Any]],
) -> pd.DataFrame:
    """Evalua cada conducta contra perfiles historicos comparables."""
    work = df.copy()
    periodos = _periodos_ordenados(work[cfg.columna_tiempo])
    scopes = _claves_referencia(cfg)

    # Un marco separado protege las columnas del usuario. El codigo -1 para
    # nulos nunca coincide con una categoria literal, incluido '__MISSING__'.
    ctx = pd.DataFrame({c: pd.factorize(work[c].astype("string"))[0]
                        for c in cfg.context_vars}, index=work.index)

    for behavior in cfg.behavior_vars:
        valores = pd.to_numeric(work[behavior], errors="coerce")
        for scope, cols_originales in scopes:
            cols = cols_originales
            col_nll = _nombre("kde", behavior, scope, "neglog_density")
            col_tail = _nombre("kde", behavior, scope, "tail_rarity")
            salida_nll = pd.Series(np.nan, index=work.index, dtype=float)
            salida_tail = pd.Series(np.nan, index=work.index, dtype=float)

            for pos_periodo, periodo in enumerate(periodos):
                if pos_periodo == 0:
                    continue
                mask_actual = work[cfg.columna_tiempo].eq(periodo) & valores.notna()
                mask_pasado = work[cfg.columna_tiempo].isin(periodos[:pos_periodo]) & valores.notna()
                if not mask_actual.any() or not mask_pasado.any():
                    continue

                actual = ctx.loc[mask_actual, cols]
                pasado = ctx.loc[mask_pasado, cols]
                grupos_pasados = pasado.groupby(cols, dropna=False, sort=False).groups

                for clave, idx_actual_local in actual.groupby(cols, dropna=False, sort=False).groups.items():
                    clave_tuple = clave if isinstance(clave, tuple) else (clave,)
                    # pandas usa escalar para groupby de una columna y tupla para varias.
                    lookup = clave_tuple if len(cols) > 1 else clave_tuple[0]
                    idx_ref = grupos_pasados.get(lookup)
                    if idx_ref is None or len(idx_ref) < cfg.min_group_size:
                        continue
                    idx_ref = pd.Index(idx_ref)
                    if cfg.kde_max_reference_rows > 0 and len(idx_ref) > cfg.kde_max_reference_rows:
                        posiciones = np.linspace(0, len(idx_ref) - 1, cfg.kde_max_reference_rows, dtype=int)
                        idx_ref = idx_ref[posiciones]

                    x_ref = valores.loc[idx_ref].to_numpy(dtype=float)
                    z_ref, mediana, escala = _escalado_robusto(x_ref, cfg.epsilon)
                    p_ref = work.loc[idx_ref, cfg.columna_tiempo].to_numpy()
                    bw = _elegir_bandwidth(z_ref, p_ref, cfg)
                    kde = KernelDensity(kernel=cfg.kernel_type, bandwidth=bw).fit(z_ref[:, None])

                    idx_eval = pd.Index(idx_actual_local)
                    z_eval = (valores.loc[idx_eval].to_numpy(dtype=float) - mediana) / escala
                    log_density = kde.score_samples(z_eval[:, None])
                    ordenados = np.sort(z_ref)
                    rango = np.searchsorted(ordenados, z_eval, side="right")
                    p_izq = rango / len(ordenados)
                    p_der = 1.0 - np.searchsorted(ordenados, z_eval, side="left") / len(ordenados)
                    p_dos_colas = np.minimum(1.0, 2.0 * np.minimum(p_izq, p_der))
                    # Kernels de soporte compacto devuelven -inf fuera del
                    # soporte; epsilon mantiene la feature finita y auditable.
                    salida_nll.loc[idx_eval] = -np.maximum(log_density, np.log(cfg.epsilon))
                    salida_tail.loc[idx_eval] = -np.log(np.maximum(p_dos_colas, cfg.epsilon))

            work[col_nll] = salida_nll
            work[col_tail] = salida_tail
            ref_texto = "+".join(cols_originales)
            catalogo.extend([
                _fila_catalogo(
                    col_nll, "KDE", behavior, ref_texto,
                    "-log densidad KDE contra el mismo perfil usando solo meses anteriores.",
                ),
                _fila_catalogo(
                    col_tail, "KDE", behavior, ref_texto,
                    "-log probabilidad empirica bilateral contra el mismo perfil historico.",
                ),
            ])

    return work


def _evaluar_estabilidad(
    df: pd.DataFrame, cfg: ConfigPipeline, columnas: list[str],
) -> pd.DataFrame:
    """Resume cobertura y deriva temporal de cada feature generada."""
    filas: list[dict[str, Any]] = []
    for col in columnas:
        agrupado = df.groupby(cfg.columna_tiempo, sort=True)[col]
        for periodo, serie in agrupado:
            valida = pd.to_numeric(serie, errors="coerce").dropna()
            filas.append({
                "columna": col,
                "periodo": periodo,
                "n": int(len(serie)),
                "n_validos": int(len(valida)),
                "pct_nulos": float(serie.isna().mean()),
                "media": float(valida.mean()) if len(valida) else np.nan,
                "std": float(valida.std(ddof=0)) if len(valida) else np.nan,
                "p95": float(valida.quantile(0.95)) if len(valida) else np.nan,
            })
    return pd.DataFrame(filas)


def ejecutar(df: pd.DataFrame, cfg: ConfigPipeline) -> ResultadoFeatureEngineering:
    """Ejecuta FE opcional; con el interruptor apagado devuelve una copia sin cambios."""
    if not cfg.usar_feature_engineering:
        LOGGER.info("Feature engineering de anomalias OMITIDO (usar_feature_engineering=False).")
        return ResultadoFeatureEngineering(df.copy(), pd.DataFrame(), pd.DataFrame())

    _validar_columnas(df, cfg)
    catalogo: list[dict[str, Any]] = []
    LOGGER.info(
        "Feature engineering: %d behavior_vars, %d context_vars, modo=%s, kernel=%s.",
        len(cfg.behavior_vars), len(cfg.context_vars), cfg.reference_mode, cfg.kernel_type,
    )
    enriquecido = _agregar_temporales(df, cfg, catalogo)
    enriquecido = _agregar_kde(enriquecido, cfg, catalogo)
    catalogo_df = pd.DataFrame(catalogo).drop_duplicates("columna").reset_index(drop=True)
    catalogo_df["pct_nulos"] = catalogo_df["columna"].map(enriquecido.isna().mean())
    estabilidad = _evaluar_estabilidad(enriquecido, cfg, catalogo_df["columna"].tolist())
    LOGGER.info(
        "Feature engineering completado: %d columnas nuevas (%d temporales, %d KDE).",
        len(catalogo_df), int((catalogo_df["familia"] == "TEMPORAL").sum()),
        int((catalogo_df["familia"] == "KDE").sum()),
    )
    return ResultadoFeatureEngineering(enriquecido, catalogo_df, estabilidad)


def construir_ablacion(catalogo: pd.DataFrame, seleccionadas: list[str]) -> pd.DataFrame:
    """Ablacion estructural: supervivencia de originales, temporales y KDE."""
    if catalogo.empty:
        return pd.DataFrame()
    generadas = set(catalogo["columna"])
    seleccion = set(seleccionadas)
    filas = []
    for familia in ("TEMPORAL", "KDE"):
        cols = set(catalogo.loc[catalogo["familia"] == familia, "columna"])
        n_sel = len(cols & seleccion)
        filas.append({
            "familia": familia,
            "generadas": len(cols),
            "seleccionadas": n_sel,
            "descartadas": len(cols) - n_sel,
            "tasa_supervivencia": n_sel / len(cols) if cols else np.nan,
        })
    filas.append({
        "familia": "TOTAL_FE",
        "generadas": len(generadas),
        "seleccionadas": len(generadas & seleccion),
        "descartadas": len(generadas - seleccion),
        "tasa_supervivencia": len(generadas & seleccion) / len(generadas) if generadas else np.nan,
    })
    return pd.DataFrame(filas)


def preparar_para_modelos(
    df: pd.DataFrame,
    cfg: ConfigPipeline,
    variables_seleccionadas: list[str],
) -> ResultadoPreparacionModelos:
    """Imputa, codifica y escala con parametros aprendidos solo en TRAIN.

    Las categoricas usan frequency encoding aprendido en TRAIN. Todas las
    columnas se escalan por mediana/IQR de TRAIN; este contrato numerico sirve
    tanto para Isolation Forest como para la entrada tabular de un VAE.
    """
    if not (cfg.usar_feature_engineering and cfg.preparar_modelos_anomalia):
        return ResultadoPreparacionModelos(pd.DataFrame(), pd.DataFrame(), None)
    if not variables_seleccionadas:
        LOGGER.warning("No se prepara matriz IF/VAE: no hay variables seleccionadas.")
        return ResultadoPreparacionModelos(pd.DataFrame(), pd.DataFrame(), None)

    periodos = _periodos_ordenados(df[cfg.columna_tiempo])
    if len(periodos) <= cfg.meses_holdout_anomalia:
        raise ErrorFeatureEngineering(
            f"Se requieren mas de {cfg.meses_holdout_anomalia} periodos para separar TRAIN/HOLDOUT; "
            f"solo hay {len(periodos)}."
        )
    holdout = set(periodos[-cfg.meses_holdout_anomalia:])
    mask_train = ~df[cfg.columna_tiempo].isin(holdout)
    matriz_x = pd.DataFrame(index=df.index)
    esquema: list[dict[str, Any]] = []

    for col in variables_seleccionadas:
        serie = df[col]
        train = serie.loc[mask_train]
        if pd.api.types.is_numeric_dtype(serie):
            train_num = pd.to_numeric(train, errors="coerce")
            imputacion = float(train_num.median()) if train_num.notna().any() else 0.0
            numerica = pd.to_numeric(serie, errors="coerce").fillna(imputacion).astype(float)
            metodo = "NUMERICA_MEDIANA"
        else:
            train_txt = train.astype("string").fillna("__MISSING__")
            frecuencias = train_txt.value_counts(normalize=True).to_dict()
            numerica = serie.astype("string").fillna("__MISSING__").map(frecuencias).fillna(0.0).astype(float)
            imputacion = "__MISSING__/0 para categoria no vista"
            metodo = "CATEGORICA_FRECUENCIA"

        centro = float(numerica.loc[mask_train].median())
        q25, q75 = numerica.loc[mask_train].quantile([0.25, 0.75])
        escala = float(q75 - q25)
        if not np.isfinite(escala) or escala <= cfg.epsilon:
            escala = float(numerica.loc[mask_train].std(ddof=0))
        if not np.isfinite(escala) or escala <= cfg.epsilon:
            escala = 1.0
        matriz_x[col] = (numerica - centro) / escala
        esquema.append({
            "columna": col,
            "transformacion": metodo,
            "imputacion_train": imputacion,
            "centro_train": centro,
            "escala_iqr_train": escala,
            "nulos_salida": int(matriz_x[col].isna().sum()),
        })

    roles = [cfg.columna_id, cfg.columna_tiempo]
    if cfg.columna_target in df.columns:
        roles.append(cfg.columna_target)
    salida = df[roles].copy()
    salida.insert(len(roles), "split_temporal", np.where(mask_train, "TRAIN", "HOLDOUT"))
    salida = pd.concat([salida, matriz_x], axis=1)
    ruta = cfg.ruta_matriz_anomalias_efectiva
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(ruta, index=False, sep=cfg.csv_sep, encoding=cfg.csv_encoding)
    LOGGER.info(
        "Matriz IF/VAE exportada: %s (%d TRAIN, %d HOLDOUT, %d variables).",
        ruta.resolve(), int(mask_train.sum()), int((~mask_train).sum()), len(variables_seleccionadas),
    )
    return ResultadoPreparacionModelos(salida, pd.DataFrame(esquema), str(ruta.resolve()))
