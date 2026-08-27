"""
fase2_no_supervisado.py
========================

FASE 2 (RAMA ALTERNATIVA, SIN TARGET) - Relevancia estructural no supervisada.

Se activa automaticamente cuando ``cfg.columna_target`` no existe en el
dataset cargado (lo detecta ``validaciones.target_disponible`` y lo propaga
``pipeline.py``). Reemplaza la fase bivariada supervisada (IV/Gini, que exigen
una etiqueta) por dos medidas que NO la requieren, elegidas porque son las que
la literatura de seleccion de variables no supervisada y de deteccion de
anomalias identifica como mas defendibles quando el consumidor final es un
modelo como Isolation Forest o un autoencoder variacional (VAE):

1. **Laplacian Score** (He, Cai y Niyogi, NeurIPS 2005): mide si una variable
   es consistente con la estructura de vecindad local de los datos (el
   "manifold" que forman todas las variables juntas). Es, literalmente, el
   metodo de referencia mas citado para seleccion de variables no supervisada
   por filtro (ver la revision de Li et al., ACM Computing Surveys, 2018,
   "Feature Selection: A Data Perspective", seccion de metodos espectrales).
   Se contrasta contra un piso de ruido estimado por PERMUTACION, exactamente
   con la misma logica que el piso de ruido del IV en la fase bivariada
   supervisada (ver ``metricas.piso_ruido_iv``): si el score real de una
   variable no es distinguible del que obtendria una version barajada al
   azar, la variable no aporta estructura.

2. **Dispersion robusta / entropia** (informativo, no eliminatorio por si
   solo): las anomalias, por definicion, viven en las colas de la
   distribucion marginal de una variable (Aggarwal, *Outlier Analysis*,
   2a ed., Springer 2016). Una variable con colas pesadas (curtosis alta) o,
   en categoricas, con alta entropia, tiene mas "espacio" donde un valor
   inusual pueda manifestarse. Se usa como criterio de RANKING (desempate en
   la fase 3), no de exclusion dura: la fase 1 ya elimino las variables sin
   dispersion alguna, asi que este componente afina el orden entre las que
   sobrevivieron, no decide sobrevivencia.

Por que NO se replica Boruta en esta rama
------------------------------------------
Boruta compara la importancia de cada variable real contra "shadow features"
usando un Random Forest ENTRENADO CONTRA UN TARGET. Sin target no hay nada
contra que entrenar ese bosque, y no existe una forma honesta de improvisar
uno sin inventar una variable (lo que el usuario pidio explicitamente evitar).
Por eso esta rama tiene EXACTAMENTE tres fases -como la rama supervisada sin
contar Boruta, que siempre fue opcional-: univariado (identico, no usa
target), esta fase 2 alternativa, y multivariado (identico, no usa target).

Por que la redundancia importa MAS aqui, no menos
---------------------------------------------------
Sin el filtro de poder predictivo marginal (IV/Gini) que en la rama
supervisada ya elimina buena parte del ruido, la fase 3 (correlacion/VIF) es
la unica barrera que queda contra variables duplicadas o casi duplicadas. Eso
es especialmente critico para los dos modelos objetivo:

* **Isolation Forest** (Liu, Ting y Zhou, ICDM 2008) selecciona al azar la
  variable de cada corte: si 10 de 50 variables son copias correlacionadas de
  la misma senal, esa senal recibe efectivamente 10 veces mas probabilidad de
  ser elegida en cada corte que una variable unica e igual de informativa,
  sesgando el bosque sin que el algoritmo lo sepa. Ademas, en alta dimension
  con variables irrelevantes, los outliers solo son visibles en SUBESPACIOS
  de baja dimension (Aggarwal y Yu, SIGMOD 2001): arrastrar variables
  redundantes o irrelevantes diluye exactamente esos subespacios.
* **VAE** (An y Cho, 2015, *Variational Autoencoder based Anomaly Detection
  using Reconstruction Probability*): la perdida de reconstruccion (tipicamente
  MSE) se reparte entre todas las variables de entrada; un grupo de variables
  correlacionadas "vota" varias veces por la misma senal en esa perdida,
  sesgando al encoder a reconstruirla bien mientras ignora variables raras
  pero mas informativas sobre el fenomeno anomalo.

Referencia general del panorama (clasico y profundo) de deteccion de
anomalias, para quien quiera el contexto completo mas alla de esta fase:
Chandola, Banerjee y Kumar, *Anomaly Detection: A Survey*, ACM Computing
Surveys 41(3), 2009; Pang, Shen, Cao y van den Hengel, *Deep Learning for
Anomaly Detection: A Review*, ACM Computing Surveys 54(2), 2021; Ruff et al.,
*A Unifying Review of Deep and Shallow Anomaly Detection*, Proceedings of the
IEEE 109(5), 2021.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ConfigPipeline
from .logging_utils import obtener_logger
from .metricas import (
    construir_matriz_numerica,
    entropia_normalizada,
    laplacian_score_con_piso_ruido,
    normalizar,
)

LOGGER = obtener_logger("fase2ns")


def ejecutar(
    df: pd.DataFrame,
    cfg: ConfigPipeline,
    tipos: dict[str, str],
    sobrevivientes: list[str],
    diagnostico: pd.DataFrame,
) -> pd.DataFrame:
    """Ejecuta la fase 2 no supervisada.

    Parameters
    ----------
    diagnostico
        Tabla de la fase 0 (una fila por columna original). Se reutiliza la
        curtosis ya calculada ahi para el componente de dispersion, en vez de
        recalcularla: evita duplicar computo y mantiene una unica fuente de
        verdad para los estadisticos descriptivos del proyecto.

    Returns
    -------
    pd.DataFrame
        Con ``columna``, ``score_compuesto`` (mismo nombre que en la rama
        supervisada, para que la fase 3 pueda consumir cualquiera de las dos
        sin distinguir el modo) y las columnas propias del metodo.
    """
    LOGGER.info("=" * 78)
    LOGGER.info("FASE 2 (NO SUPERVISADA) - RELEVANCIA ESTRUCTURAL SIN TARGET")
    LOGGER.info(
        "Variables a evaluar=%d | k_vecinos=%d | permutaciones=%d | alpha=%.3f | "
        "pesos laplaciano/dispersion=%.2f/%.2f",
        len(sobrevivientes), cfg.laplacian_k_vecinos, cfg.laplacian_n_permutaciones,
        cfg.alpha_ruido_laplaciano, cfg.peso_laplaciano, cfg.peso_dispersion,
    )
    LOGGER.info("=" * 78)

    if not sobrevivientes:
        LOGGER.warning("No hay variables que evaluar en la fase 2 no supervisada.")
        return pd.DataFrame()

    if len(sobrevivientes) < 2:
        LOGGER.warning(
            "Solo hay %d variable(s) superviviente(s) de la fase 1: el Laplacian Score "
            "necesita un grafo de vecindad construido con al menos 2 variables. "
            "Se retiene la unica variable sin poder contrastarla.",
            len(sobrevivientes),
        )
        col = sobrevivientes[0]
        return pd.DataFrame([{
            "columna": col, "tipo_inferido": tipos.get(col, "?"),
            "laplacian_score": np.nan, "piso_ruido_laplaciano": np.nan,
            "p_valor_estructura": np.nan, "dispersion_bruta": np.nan,
            "laplacian_normalizado": np.nan, "dispersion_normalizada": np.nan,
            "score_compuesto": 1.0, "flg_no_evaluable": 1, "flg_sin_estructura": 0,
            "flg_exclusion": 0, "flg_seleccionada_no_supervisada": 1,
            "motivo_exclusion_no_supervisada": "",
            "observacion": "Unica variable superviviente; no evaluable contra un grafo propio.",
        }])

    # ------------------------------------------------------------------
    # 1. Matriz numerica conjunta (misma utilidad que usa Boruta) y grafo
    #    de vecindad + Laplacian Score con piso de ruido por permutacion.
    # ------------------------------------------------------------------
    X, notas_matriz = construir_matriz_numerica(df, sobrevivientes, tipos)

    LOGGER.info("Calculando Laplacian Score sobre %d variables (grafo de %d filas)...",
                len(sobrevivientes), min(len(X), cfg.laplacian_max_filas or len(X)))

    if cfg.bonferroni_ruido_laplaciano:
        alpha_bonf = cfg.alpha_ruido_laplaciano / max(len(sobrevivientes), 1)
        resolucion = 1.0 / max(cfg.laplacian_n_permutaciones, 1)
        if resolucion > alpha_bonf:
            LOGGER.warning(
                "bonferroni_ruido_laplaciano=True con alpha efectivo=%.5f, pero "
                "laplacian_n_permutaciones=%d solo resuelve p-valores hasta %.5f: el test "
                "no tiene resolucion suficiente para aplicar ese umbral con precision. "
                "Suba laplacian_n_permutaciones a >= %d para una estimacion confiable.",
                alpha_bonf, cfg.laplacian_n_permutaciones, resolucion,
                int(np.ceil(1.0 / alpha_bonf)),
            )

    tabla_laplaciano = laplacian_score_con_piso_ruido(
        X,
        k_vecinos=cfg.laplacian_k_vecinos,
        n_permutaciones=cfg.laplacian_n_permutaciones,
        alpha=cfg.alpha_ruido_laplaciano,
        semilla=cfg.semilla,
        max_filas=cfg.laplacian_max_filas,
        bonferroni_n=len(sobrevivientes) if cfg.bonferroni_ruido_laplaciano else 1,
    )

    # ------------------------------------------------------------------
    # 2. Dispersion robusta / entropia (RANKING, no exclusion; ver docstring).
    #    Numericas -> |curtosis| ya calculada en la fase 0 (evita recomputo).
    #    Categoricas -> entropia normalizada (la curtosis no esta definida
    #    sobre categorias sin orden).
    # ------------------------------------------------------------------
    curtosis_por_col = diagnostico.set_index("columna")["curtosis"] if "curtosis" in diagnostico.columns else pd.Series(dtype=float)
    dispersion: dict[str, float] = {}
    for col in sobrevivientes:
        tipo = tipos.get(col, "CATEGORICA")
        if tipo in ("NUMERICA", "BOOLEANA"):
            curt = curtosis_por_col.get(col, np.nan)
            # Solo la cola PESADA (curtosis positiva/leptocurtica) es indicio de
            # potencial de outlier; una distribucion platicurtica (curtosis
            # negativa) no aporta menos estructura, aporta un tipo distinto que
            # este proyecto no penaliza pero tampoco premia como "outlier-prone".
            dispersion[col] = max(float(curt), 0.0) if pd.notna(curt) else np.nan
        else:
            dispersion[col] = entropia_normalizada(df[col])

    # ------------------------------------------------------------------
    # 3. Score compuesto: mismo patron que score_compuesto() de la rama
    #    supervisada (normalizar + ponderar), pero con Laplacian (invertido:
    #    menor score real = mayor relevancia) y dispersion/entropia.
    # ------------------------------------------------------------------
    tabla = tabla_laplaciano.copy()
    tabla["dispersion_bruta"] = tabla["columna"].map(dispersion)
    tabla["laplacian_normalizado"] = 1.0 - normalizar(tabla["laplacian_score"], "minmax")
    tabla["dispersion_normalizada"] = normalizar(tabla["dispersion_bruta"], "minmax")

    # Si la dispersion no es calculable (categorica sin variacion evaluable),
    # el score cae integramente sobre el Laplacian en vez de penalizar a la
    # variable por carecer de un componente que no le aplica.
    tiene_ambos = tabla["laplacian_normalizado"].notna() & tabla["dispersion_normalizada"].notna()
    tabla["score_compuesto"] = np.where(
        tiene_ambos,
        cfg.peso_laplaciano * tabla["laplacian_normalizado"].fillna(0)
        + cfg.peso_dispersion * tabla["dispersion_normalizada"].fillna(0),
        tabla["laplacian_normalizado"],
    )
    tabla["peso_laplaciano"] = cfg.peso_laplaciano
    tabla["peso_dispersion"] = cfg.peso_dispersion
    tabla["tipo_inferido"] = tabla["columna"].map(tipos)
    tabla["nota_codificacion"] = tabla["columna"].map(notas_matriz)

    # ------------------------------------------------------------------
    # 4. Regla de exclusion: unica y clara -> no supera el piso de ruido.
    #    A diferencia de la rama supervisada (IV Y Gini), aqui hay una sola
    #    prueba con evidencia estadistica formal (el test de permutacion);
    #    la dispersion es solo de ranking, por eso no participa del "Y".
    # ------------------------------------------------------------------
    tabla["flg_no_evaluable"] = tabla["laplacian_score"].isna().astype(int)
    tabla["flg_sin_estructura"] = (
        (tabla["flg_supera_ruido"] == 0) & (tabla["flg_no_evaluable"] == 0)
    ).astype(int)

    motivos, excluir = [], []
    for _, f in tabla.iterrows():
        razones = []
        if f["flg_no_evaluable"]:
            razones.append("Laplacian Score no calculable (varianza nula tras estandarizar).")
        elif f["flg_sin_estructura"]:
            razones.append(
                f"Laplacian Score={f['laplacian_score']:.4f} no distinguible del ruido de "
                f"permutacion (piso={f['piso_ruido_laplaciano']:.4f}, "
                f"p={f['p_valor_estructura']:.4f} >= alpha={cfg.alpha_ruido_laplaciano}): "
                "la variable no es mas consistente con la estructura de vecindad de los "
                "datos que una version barajada al azar de si misma."
            )
        excluir.append(int(bool(razones)))
        motivos.append("; ".join(razones))
    tabla["flg_exclusion"] = excluir
    tabla["motivo_exclusion_no_supervisada"] = motivos

    if cfg.top_n_no_supervisado > 0:
        vivas = tabla[tabla["flg_exclusion"] == 0].sort_values("score_compuesto", ascending=False)
        fuera = vivas.iloc[cfg.top_n_no_supervisado:]["columna"]
        mask = tabla["columna"].isin(fuera)
        tabla.loc[mask, "flg_exclusion"] = 1
        tabla.loc[mask, "motivo_exclusion_no_supervisada"] += (
            f"; fuera del top-{cfg.top_n_no_supervisado} por score compuesto"
        )

    tabla["flg_seleccionada_no_supervisada"] = 1 - tabla["flg_exclusion"]
    tabla = tabla.sort_values(
        "score_compuesto", ascending=False, na_position="last"
    ).reset_index(drop=True)
    tabla["ranking_score"] = np.arange(1, len(tabla) + 1)

    n_out = int(tabla["flg_exclusion"].sum())
    LOGGER.info("Variables evaluadas          : %d", len(tabla))
    LOGGER.info("Sin estructura (piso ruido)  : %d", int(tabla["flg_sin_estructura"].sum()))
    LOGGER.info("Excluidas en la fase 2 (NS)  : %d", n_out)
    LOGGER.info("Sobrevivientes de la fase 2  : %d", len(tabla) - n_out)
    if len(tabla):
        top = tabla.head(5)[["columna", "laplacian_score", "dispersion_bruta", "score_compuesto"]]
        LOGGER.info("Top 5 por score compuesto:\n%s", top.to_string(index=False))

    return tabla


def obtener_sobrevivientes(tabla: pd.DataFrame) -> list[str]:
    """Variables que pasan a la fase multivariada."""
    if tabla.empty:
        return []
    return tabla.loc[tabla["flg_exclusion"] == 0, "columna"].tolist()
