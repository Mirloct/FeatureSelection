"""
config.py
=========

FUENTE UNICA DE VERDAD de la configuracion del proyecto.

Metadata
--------
Data sources / inputs: ``config.yaml`` y overrides de ``run_pipeline.py``.
Created: 2026-07-26
Last modified: 2026-10-04
Changelog:
- 2026-10-02: opciones centralizadas y validacion de agrupacion de puestos.
- 2026-10-02: separa columnas conservadas sin evaluar y excluidas totalmente.
- 2026-10-02: documenta exclusiones manuales como columnas conservadas al final.
- 2026-10-01: se centralizo la configuracion de feature engineering temporal,
  KDE condicional y preparacion time-safe para Isolation Forest/VAE.

- 2026-10-02: valores por defecto exclusivamente en config.yaml; carga estricta
  y construccion directa desde la misma fuente, sin defaults duplicados.
- 2026-10-04: valida parametros KDE finitos y ventanas enteras sin truncar.
- 2026-10-04: comentarios de resolucion Monte Carlo con correccion +1.

Edite nombres de columnas, rutas y valores en config.yaml.
Precedencia: YAML central < YAML alternativo < flags del CLI.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .logging_utils import obtener_logger

LOGGER = obtener_logger("config")
CONFIG_PREDETERMINADA = Path(__file__).resolve().parents[2] / "config.yaml"


# ---------------------------------------------------------------------------
# Excepcion de dominio
# ---------------------------------------------------------------------------
class ErrorConfiguracion(ValueError):
    """Se lanza cuando la configuracion es invalida o inconsistente."""


def _positivo_finito(valor: Any) -> bool:
    """Valida numeros configurados sin propagar errores de conversion."""
    try:
        return not isinstance(valor, bool) and math.isfinite(float(valor)) and float(valor) > 0
    except (ValueError, TypeError, OverflowError):
        return False


def _ventana_valida(valor: Any) -> bool:
    try:
        return _positivo_finito(valor) and float(valor) >= 2 and float(valor).is_integer()
    except (ValueError, TypeError, OverflowError):
        return False


# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
@dataclass(init=False)
class ConfigPipeline:
    """Parametros completos del pipeline de seleccion de variables.

    Se divide en bloques: entradas obligatorias, umbrales por fase y opciones
    de ejecucion. Todos los umbrales son parametrizables; los valores por
    defecto son los justificados en la documentacion (`docs/documentacion.html`).
    """

    # =====================================================================
    # BLOQUE A. Entradas obligatorias del proceso
    # =====================================================================
    ruta_dataset: str
    columna_target: str
    columna_id: str
    columna_tiempo: str
    usar_boruta: bool
    ruta_salida_excel: str

    #: Columnas adicionales sin evaluar, conservadas al final del dataset exportado
    #: (identificadores secundarios, campos de auditoria, llaves foraneas...).
    columnas_conservadas: list[str]
    #: Columnas eliminadas antes del analisis; no se exportan.
    columnas_excluidas: list[str]

    #: Si True, ademas del Excel de bitacora se exporta un dataset "listo para
    #: modelar": id + tiempo + target + seleccionadas + exclusiones manuales.
    #: Las seleccionadas son las variables que superaron las
    #: tres fases obligatorias (univariada, bivariada y multivariada).
    exportar_dataset_final: bool
    #: Ruta del dataset final. Si se deja vacio, se deriva automaticamente junto
    #: a `ruta_salida_excel` como "<mismo_nombre>_dataset_final.<formato>".
    ruta_dataset_final: str
    #: Formato de escritura del dataset final: "csv" | "parquet".
    formato_dataset_final: str

    # =====================================================================
    # BLOQUE B. Fase 1 - Univariado
    # =====================================================================
    #: Umbral PRINCIPAL de (%ceros + %nulos). >= a esto -> se elimina.
    #: 0.95 = la variable es constante en el 95% de las filas: aporta senal
    #: solo en <=5% de la muestra y en panel eso suele ser ruido de un puñado
    #: de entidades.
    umbral_ceros_nulos: float
    #: Umbral ALTERNO, mas conservador. Se activa con `usar_umbral_alterno`.
    umbral_ceros_nulos_alterno: float
    usar_umbral_alterno: bool

    #: Desviacion estandar por debajo de la cual se considera constante.
    umbral_std_minimo: float
    #: Coeficiente de variacion (std/|media|) minimo. Mide dispersion RELATIVA:
    #: una std de 0.01 es despreciable si la media es 1e6, pero enorme si la
    #: media es 0.001. Por eso no basta con mirar la std absoluta.
    umbral_cv_minimo: float
    #: Proporcion maxima admisible del valor/categoria mas frecuente.
    #: >= 0.99 -> una sola categoria domina y la variable es casi constante.
    umbral_dominancia: float
    #: Zona gris de dominancia, SOLO para CATEGORICAS: aviso (no elimina)
    #: cuando la categoria mas frecuente cubre entre este umbral y
    #: `umbral_dominancia`. Una categoria que concentra ~90% de la masa no es
    #: "casi constante" (eso exige 99%), pero codificarla (one-hot, WOE) deja
    #: a la clase minoritaria con muy pocas observaciones: el modelo puede
    #: sesgarse hacia la clase dominante o sobreajustar la minoritaria. Debe
    #: ser < `umbral_dominancia`.
    umbral_dominancia_aviso: float
    #: Cantidad de categorias distintas por encima de la cual una CATEGORICA
    #: se marca como de "cardinalidad alta" (aviso, no elimina): el one-hot
    #: deja de ser practico (explosion dimensional, columnas casi vacias) y
    #: conviene agrupar categorias raras o usar una codificacion que no
    #: multiplique columnas (la fase 2 ya agrupa en __OTROS__ via
    #: `max_categorias`; la rama no supervisada ya usa codigo ordinal por
    #: frecuencia en vez de one-hot). Este umbral es mas bajo que
    #: `max_categorias` a proposito: avisa antes de que el agrupamiento entre
    #: a actuar, para que la decision de fondo (¿esta variable es realmente
    #: util con esta granularidad?) se tome con informacion, no en silencio.
    umbral_alta_cardinalidad: int
    #: IQR (p75-p25) por debajo del cual se marca "percentiles comprimidos".
    umbral_iqr_minimo: float
    #: Numero minimo de valores unicos para considerar evaluable una variable.
    minimo_valores_unicos: int

    # =====================================================================
    # BLOQUE B2. Fase 1B - Agrupacion de categoricas por similitud de nombre
    # =====================================================================
    # Comun a ambas ramas (corre antes del fork, no usa el target). Ataca el
    # caso que ni la fase 1 (dominancia) ni el agrupamiento por frecuencia de
    # la fase 2 (`max_categorias`) resuelven bien: una categorica genuinamente
    # dispersa (ninguna categoria domina) pero con demasiados niveles para un
    # one-hot util. Ver `fase1b_agrupacion_categorica.py` y
    # `metricas.agrupar_categoria_por_similitud_nombre`.
    #: Interruptor maestro de la fase.
    usar_agrupacion_categorica_nombre: bool
    #: Cardinalidad minima para activar el CLUSTERING (accion, no solo aviso).
    #: Deliberadamente mayor que `umbral_alta_cardinalidad` (20, que solo
    #: avisa en la fase 1): ese umbral bajo existe para que se vea el aviso
    #: aunque no se actue; este es el umbral en el que SI se actua.
    umbral_cardinalidad_clustering: int
    #: Tope superior del barrido de k explorado por silueta. Se acota ademas
    #: a `n_unicos - 1` por columna (no tiene sentido proponer mas grupos que
    #: categorias menos uno). 30 es un techo generoso frente al piso de
    #: activacion (100 categorias): incluso en el peor caso reduce la
    #: cardinalidad a menos de un tercio.
    max_k_agrupacion_categorica: int

    # =====================================================================
    # BLOQUE C. Fase 2 - Bivariado (IV / Gini)
    # =====================================================================
    #: Numero de bins objetivo para la discretizacion por cuantiles (WOE).
    n_bins: int
    #: Fraccion minima de la muestra por bin. Bins mas chicos se fusionan:
    #: un bin con 5 observaciones produce un WOE inestable y un IV inflado.
    min_prop_bin: float
    #: Cardinalidad maxima de una categorica; el resto se agrupa en "OTROS".
    max_categorias: int
    #: Correccion de continuidad (Haldane-Anscombe) para evitar log(0) en WOE.
    correccion_woe: float

    #: Pesos del score compuesto. Por defecto balanceado 50/50.
    peso_gini: float
    peso_iv: float
    #: Normalizacion previa a la ponderacion: "minmax" | "rank".
    metodo_normalizacion: str

    #: Pisos de poder predictivo. Regla de exclusion: se descarta la variable
    #: solo si falla en AMBAS metricas (IV bajo Y Gini bajo), para no penalizar
    #: variables que una de las dos metricas capta mejor.
    umbral_iv_minimo: float
    umbral_gini_minimo: float
    #: Piso opcional sobre el score compuesto normalizado (0 = desactivado).
    umbral_score_minimo: float

    #: Piso de ruido estadistico. Los umbrales fijos (0.02 / 0.05) no dependen
    #: del tamano de la muestra ni del numero de bins, pero el IV espurio de una
    #: variable aleatoria SI depende de ambos. Con esto activado, el umbral
    #: efectivo es el MAYOR entre el fijo y el piso de ruido calculado.
    usar_piso_ruido: bool
    #: Nivel de significancia del contraste contra la hipotesis de irrelevancia.
    alpha_ruido: float
    bonferroni_ruido: bool
    umbral_iv_sospechoso: float
    excluir_sospecha_fuga: bool
    top_n_bivariado: int
    umbral_psi: float
    excluir_por_inestabilidad: bool

    # Rama sin target: Laplacian Score y ranking por dispersion.
    laplacian_k_vecinos: int
    laplacian_max_filas: int
    #: Permutaciones Monte Carlo; p minimo y resolucion = 1/(n+1).
    laplacian_n_permutaciones: int
    alpha_ruido_laplaciano: float
    #: Bonferroni: alpha/n_variables. Requiere 1/(permutaciones+1) < alpha efectivo.
    bonferroni_ruido_laplaciano: bool
    #: Pesos del score no supervisado compuesto (Laplacian + dispersion/entropia).
    peso_laplaciano: float
    peso_dispersion: float
    #: Si >0, conserva solo las N mejores por score no supervisado.
    top_n_no_supervisado: int

    # =====================================================================
    # BLOQUE D. Fase 3 - Multivariado
    # =====================================================================
    #: Umbral de asociacion absoluta para declarar redundancia.
    umbral_correlacion: float
    #: Metodo para pares numerico-numerico: "spearman" | "pearson".
    metodo_correlacion: str
    #: VIF por encima del cual se marca multicolinealidad severa.
    umbral_vif: float
    #: Si True, ademas de marcar, el VIF elimina iterativamente.
    excluir_por_vif: bool

    # =====================================================================
    # BLOQUE E. Fase 4 - Boruta (opcional)
    # =====================================================================
    #: "auto" (libreria si existe, si no nativo) | "borutapy" | "borutashap" | "nativo"
    motor_boruta: str
    boruta_n_estimadores: int
    boruta_max_iter: int
    boruta_alpha: float
    boruta_profundidad_max: int
    #: Submuestreo para acotar el costo de Boruta en paneles grandes (0 = sin limite).
    boruta_max_filas: int

    # =====================================================================
    # BLOQUE F. Ejecucion
    # =====================================================================
    semilla: int
    n_jobs: int
    autoinstalar_dependencias: bool
    ruta_log: str
    nivel_log: str
    #: Separador y encoding para datasets CSV.
    csv_sep: str
    csv_encoding: str
    csv_encodings_fallback: list[str]
    #: Si el dataset no existe, generar el panel sintetico de demostracion.
    generar_demo_si_falta: bool

    # =====================================================================
    # BLOQUE G. Feature engineering para anomalias (opcional, pre-depuracion)
    # =====================================================================
    #: Interruptor maestro. Si es False, el pipeline conserva exactamente el
    #: flujo historico y no exige que existan context_vars/behavior_vars.
    usar_feature_engineering: bool
    #: Perfil de comparacion para la KDE. Pueden ser numericas o categoricas;
    #: los valores se tratan como estratos, sin codificacion ordinal.
    context_vars: list[str]
    #: Conductas mensuales analizadas individualmente. Deben ser numericas.
    behavior_vars: list[str]
    #: Por defecto el contexto define pares comparables pero no se entrega al
    #: detector final (evita marcar perfiles demograficos como anomalias).
    incluir_context_vars_en_seleccion: bool
    #: Kernels soportados por sklearn.neighbors.KernelDensity.
    kernel_type: str
    #: "joint", "marginal" o "joint_and_marginal".
    reference_mode: str
    #: Minimo de observaciones historicas dentro del perfil comparable.
    min_group_size: int
    #: Observaciones previas de la entidad requeridas para z-scores personales.
    min_personal_history: int
    #: "time_safe_cv", "silverman" o un ancho numerico positivo.
    bandwidth_method: str | float
    #: Grilla de bandwidth sobre la escala robusta usada por time_safe_cv.
    bandwidth_grid: list[float]
    #: Ventanas (meses/observaciones) para estadisticos personales desplazados.
    temporal_windows: list[int]
    #: Estabilizador de divisiones, densidades y escalas robustas.
    epsilon: float
    #: Tope determinista de referencias por ajuste KDE (0 = sin tope).
    kde_max_reference_rows: int

    # Preparacion comun para Isolation Forest y VAE. La matriz se genera tras
    # la seleccion, con imputacion/codificacion/escalado aprendidos SOLO en el
    # tramo de entrenamiento temporal.
    preparar_modelos_anomalia: bool
    meses_holdout_anomalia: int
    ruta_matriz_anomalias: str

    # Preparacion de puestos previa al diagnostico y al contexto KDE.
    usar_estandarizacion_puesto: bool
    columnas_puesto_prioridad: list[str]
    columna_puesto_agrupado: str
    puesto_sin_dato: str

    def __init__(self, *args: Any, **overrides: Any) -> None:
        """Lee los valores del YAML central y aplica overrides del consumidor.

        Conserva los argumentos posicionales del dataclass original y crea
        listas independientes para cada instancia. No almacena defaults en cache.
        """
        nombres = [f.name for f in fields(type(self))]
        if len(args) > len(nombres):
            raise TypeError("Demasiados argumentos para ConfigPipeline.")
        posicionales = dict(zip(nombres, args))
        duplicados = posicionales.keys() & overrides.keys()
        if duplicados:
            raise TypeError(f"Argumentos repetidos: {sorted(duplicados)}")
        desconocidos = overrides.keys() - set(nombres)
        if desconocidos:
            raise TypeError(f"Argumentos desconocidos: {sorted(desconocidos)}")
        datos = _leer_yaml(CONFIG_PREDETERMINADA)
        faltantes = set(nombres) - datos.keys()
        if faltantes:
            raise ErrorConfiguracion(
                f"Faltan parametros en '{CONFIG_PREDETERMINADA}': {sorted(faltantes)}"
            )
        datos.update(posicionales)
        datos.update(overrides)
        for nombre in nombres:
            setattr(self, nombre, datos[nombre])

    # ------------------------------------------------------------------
    # Propiedades derivadas
    # ------------------------------------------------------------------
    @property
    def columnas_rol(self) -> list[str]:
        """Columnas reservadas (target, id, tiempo). Nunca son candidatas."""
        return [self.columna_target, self.columna_id, self.columna_tiempo]

    @property
    def columnas_no_candidatas(self) -> list[str]:
        """Roles reservados + exclusiones explicitas del usuario."""
        contexto = (
            list(self.context_vars)
            if self.usar_feature_engineering and not self.incluir_context_vars_en_seleccion
            else []
        )
        return self.columnas_rol + list(self.columnas_conservadas) + list(self.columnas_excluidas) + contexto

    @property
    def umbral_ceros_nulos_efectivo(self) -> float:
        """Umbral realmente aplicado en la fase 1.1 (principal o alterno)."""
        return self.umbral_ceros_nulos_alterno if self.usar_umbral_alterno else self.umbral_ceros_nulos

    @property
    def ruta_dataset_final_efectiva(self) -> Path:
        """Ruta real del dataset final, derivandola si el usuario no la fijo.

        Se deriva junto al Excel de bitacora (mismo directorio, mismo nombre
        base) para que ambos artefactos de una misma corrida queden agrupados
        y sea evidente a que bitacora corresponde cada dataset exportado.
        """
        if self.ruta_dataset_final.strip():
            return Path(self.ruta_dataset_final)
        base = Path(self.ruta_salida_excel)
        extension = ".parquet" if self.formato_dataset_final == "parquet" else ".csv"
        return base.with_name(f"{base.stem}_dataset_final{extension}")

    @property
    def ruta_matriz_anomalias_efectiva(self) -> Path:
        """Ruta de la matriz numerica time-safe para Isolation Forest/VAE."""
        if self.ruta_matriz_anomalias.strip():
            return Path(self.ruta_matriz_anomalias)
        base = Path(self.ruta_salida_excel)
        return base.with_name(f"{base.stem}_matriz_anomalias.csv")

    def rol_de(self, columna: str) -> str:
        """Clasifica una columna segun su papel en el panel."""
        if columna == self.columna_target:
            return "TARGET"
        if columna == self.columna_id:
            return "ID_ENTIDAD"
        if columna == self.columna_tiempo:
            return "TIEMPO"
        if columna in self.columnas_excluidas:
            return "EXCLUIDA_TOTAL"
        if columna in self.columnas_conservadas:
            return "CONSERVADA_MANUAL"
        if (
            self.usar_feature_engineering
            and not self.incluir_context_vars_en_seleccion
            and columna in self.context_vars
        ):
            return "CONTEXTO_FE"
        return "CANDIDATA"

    # ------------------------------------------------------------------
    # Serializacion
    # ------------------------------------------------------------------
    def a_dict(self) -> dict[str, Any]:
        """Diccionario plano (para la hoja de parametros del Excel)."""
        return asdict(self)

    def a_filas(self) -> list[dict[str, Any]]:
        """Filas parametro/valor/bloque para la bitacora."""
        bloques = {
            "ruta_dataset": "A. Entradas", "columna_target": "A. Entradas",
            "columna_id": "A. Entradas", "columna_tiempo": "A. Entradas",
            "usar_boruta": "A. Entradas", "ruta_salida_excel": "A. Entradas",
            "columnas_conservadas": "A. Entradas",
            "columnas_excluidas": "A. Entradas",
            "exportar_dataset_final": "A. Entradas", "ruta_dataset_final": "A. Entradas",
            "formato_dataset_final": "A. Entradas",
        }
        filas = []
        for f in fields(self):
            valor = getattr(self, f.name)
            if isinstance(valor, (list, dict)):
                valor = json.dumps(valor, ensure_ascii=False)
            filas.append(
                {
                    "parametro": f.name,
                    "valor": valor,
                    "bloque": bloques.get(f.name, _bloque_por_prefijo(f.name)),
                    "tipo": type(getattr(self, f.name)).__name__,
                }
            )
        return filas

    # ------------------------------------------------------------------
    # Validacion
    # ------------------------------------------------------------------
    def validar(self) -> None:
        """Valida coherencia interna. Se ejecuta ANTES de tocar el dataset.

        Raises
        ------
        ErrorConfiguracion
            Ante cualquier parametro fuera de rango o combinacion imposible.
        """
        errores: list[str] = []

        # --- Entradas obligatorias no vacias -------------------------------
        for nombre in ("ruta_dataset", "columna_target", "columna_id",
                       "columna_tiempo", "ruta_salida_excel"):
            valor = getattr(self, nombre)
            if not isinstance(valor, str) or not valor.strip():
                errores.append(f"'{nombre}' es obligatorio y debe ser un texto no vacio.")

        # --- Las tres columnas de rol deben ser distintas entre si ---------
        roles = [self.columna_target, self.columna_id, self.columna_tiempo]
        if len(set(roles)) != 3:
            errores.append(
                f"columna_target/columna_id/columna_tiempo deben ser distintas entre si; "
                f"se recibio {roles}."
            )
        # --- ...y no pueden estar en la lista de exclusion manual ----------
        choque = set(roles) & (set(self.columnas_conservadas) | set(self.columnas_excluidas))
        if choque:
            errores.append(f"Columnas de rol listadas en conservadas o excluidas: {sorted(choque)}.")

        conflicto = set(self.columnas_conservadas) & set(self.columnas_excluidas)
        if conflicto:
            errores.append(f"Columnas presentes en conservadas y excluidas: {sorted(conflicto)}.")
        conflicto_fe = set(self.columnas_excluidas) & (set(self.context_vars) | set(self.behavior_vars))
        if self.usar_feature_engineering and conflicto_fe:
            errores.append(f"Columnas de feature engineering excluidas totalmente: {sorted(conflicto_fe)}.")

        if self.usar_estandarizacion_puesto:
            if not self.columnas_puesto_prioridad or any(
                not isinstance(c, str) or not c.strip() for c in self.columnas_puesto_prioridad
            ):
                errores.append("columnas_puesto_prioridad debe contener nombres no vacios.")
            if not isinstance(self.columna_puesto_agrupado, str) or not self.columna_puesto_agrupado.strip():
                errores.append("columna_puesto_agrupado debe ser un nombre no vacio.")
            if not isinstance(self.puesto_sin_dato, str) or not self.puesto_sin_dato.strip():
                errores.append("puesto_sin_dato debe ser una etiqueta no vacia.")
            if self.columna_puesto_agrupado in set(roles) | set(self.columnas_excluidas) | set(self.behavior_vars):
                errores.append("El puesto agrupado no puede ser rol, conducta ni columna excluida.")
            if self.columna_puesto_agrupado in self.columnas_puesto_prioridad:
                errores.append("El puesto agrupado debe tener un nombre distinto de las fuentes.")

        # --- Rangos de proporciones ---------------------------------------
        for nombre in ("umbral_ceros_nulos", "umbral_ceros_nulos_alterno",
                       "umbral_dominancia", "umbral_dominancia_aviso",
                       "umbral_correlacion", "min_prop_bin"):
            valor = getattr(self, nombre)
            if not 0.0 < float(valor) <= 1.0:
                errores.append(f"'{nombre}'={valor} debe estar en el intervalo (0, 1].")

        if self.umbral_ceros_nulos_alterno > self.umbral_ceros_nulos:
            errores.append(
                "El umbral alterno debe ser MAS conservador (menor) que el principal: "
                f"{self.umbral_ceros_nulos_alterno} > {self.umbral_ceros_nulos}."
            )
        if self.umbral_dominancia_aviso >= self.umbral_dominancia:
            errores.append(
                "El umbral de aviso de dominancia debe ser MENOR que el umbral de eliminacion: "
                f"umbral_dominancia_aviso={self.umbral_dominancia_aviso} >= "
                f"umbral_dominancia={self.umbral_dominancia}."
            )
        if self.umbral_alta_cardinalidad < 2:
            errores.append(f"umbral_alta_cardinalidad={self.umbral_alta_cardinalidad} debe ser >= 2.")
        if self.umbral_cardinalidad_clustering < self.umbral_alta_cardinalidad:
            errores.append(
                "umbral_cardinalidad_clustering debe ser >= umbral_alta_cardinalidad (el aviso "
                f"debe dispararse antes que la accion): {self.umbral_cardinalidad_clustering} < "
                f"{self.umbral_alta_cardinalidad}."
            )
        if self.max_k_agrupacion_categorica < 2:
            errores.append(f"max_k_agrupacion_categorica={self.max_k_agrupacion_categorica} debe ser >= 2.")

        # --- Pesos del score compuesto ------------------------------------
        suma = self.peso_gini + self.peso_iv
        if suma <= 0:
            errores.append("peso_gini + peso_iv debe ser mayor que cero.")
        elif abs(suma - 1.0) > 1e-9:
            LOGGER.warning(
                "peso_gini + peso_iv = %.4f != 1. Se renormalizaran a %.3f / %.3f.",
                suma, self.peso_gini / suma, self.peso_iv / suma,
            )

        if self.metodo_normalizacion not in ("minmax", "rank"):
            errores.append(f"metodo_normalizacion='{self.metodo_normalizacion}' no valido (minmax|rank).")
        if self.metodo_correlacion not in ("spearman", "pearson"):
            errores.append(f"metodo_correlacion='{self.metodo_correlacion}' no valido (spearman|pearson).")
        if self.motor_boruta not in ("auto", "borutapy", "borutashap", "nativo"):
            errores.append(f"motor_boruta='{self.motor_boruta}' no valido (auto|borutapy|borutashap|nativo).")
        if self.formato_dataset_final not in ("csv", "parquet"):
            errores.append(f"formato_dataset_final='{self.formato_dataset_final}' no valido (csv|parquet).")

        # --- Feature engineering de anomalias -----------------------------
        kernels_validos = {"gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"}
        if self.kernel_type not in kernels_validos:
            errores.append(
                f"kernel_type='{self.kernel_type}' no valido; use uno de {sorted(kernels_validos)}."
            )
        if self.reference_mode not in ("joint", "marginal", "joint_and_marginal"):
            errores.append(
                f"reference_mode='{self.reference_mode}' no valido (joint|marginal|joint_and_marginal)."
            )
        if self.usar_feature_engineering and not self.behavior_vars:
            errores.append("behavior_vars no puede estar vacio cuando usar_feature_engineering=True.")
        if self.usar_feature_engineering and not self.context_vars:
            errores.append("context_vars no puede estar vacio cuando usar_feature_engineering=True.")
        if len(self.context_vars) != len(set(self.context_vars)):
            errores.append("context_vars contiene nombres duplicados.")
        if len(self.behavior_vars) != len(set(self.behavior_vars)):
            errores.append("behavior_vars contiene nombres duplicados.")
        choque_fe = set(self.context_vars) & set(self.behavior_vars)
        if choque_fe:
            errores.append(f"context_vars y behavior_vars deben ser disjuntas: {sorted(choque_fe)}.")
        roles_fe = {self.columna_id, self.columna_tiempo, self.columna_target}
        choque_roles = roles_fe & (set(self.context_vars) | set(self.behavior_vars))
        if choque_roles:
            errores.append(
                "Las columnas de rol no pueden usarse como contexto o conducta (evita identidad/fuga): "
                f"{sorted(choque_roles)}."
            )
        if self.min_group_size < 2:
            errores.append(f"min_group_size={self.min_group_size} debe ser >= 2.")
        if self.min_personal_history < 2:
            errores.append(f"min_personal_history={self.min_personal_history} debe ser >= 2.")
        if not _positivo_finito(self.epsilon):
            errores.append(f"epsilon={self.epsilon} debe ser finito y > 0.")
        if self.kde_max_reference_rows < 0:
            errores.append("kde_max_reference_rows debe ser >= 0 (0 significa sin limite).")
        if self.meses_holdout_anomalia < 1:
            errores.append("meses_holdout_anomalia debe ser >= 1.")
        if not self.temporal_windows or any(not _ventana_valida(w) for w in self.temporal_windows):
            errores.append("temporal_windows debe contener enteros >= 2.")
        if not self.bandwidth_grid or any(not _positivo_finito(x) for x in self.bandwidth_grid):
            errores.append("bandwidth_grid debe contener valores finitos y positivos.")
        if (self.bandwidth_method not in ("time_safe_cv", "silverman")
                and not _positivo_finito(self.bandwidth_method)):
            errores.append("bandwidth_method debe ser time_safe_cv, silverman o un numero finito positivo.")

        # --- Enteros positivos --------------------------------------------
        if self.n_bins < 2:
            errores.append(f"n_bins={self.n_bins} debe ser >= 2.")
        if self.max_categorias < 2:
            errores.append(f"max_categorias={self.max_categorias} debe ser >= 2.")
        if self.boruta_max_iter < 5:
            errores.append(f"boruta_max_iter={self.boruta_max_iter} debe ser >= 5.")
        if not 0 < self.boruta_alpha < 1:
            errores.append(f"boruta_alpha={self.boruta_alpha} debe estar en (0, 1).")

        # --- Rama no supervisada (Laplacian Score) -------------------------
        if self.laplacian_k_vecinos < 2:
            errores.append(f"laplacian_k_vecinos={self.laplacian_k_vecinos} debe ser >= 2.")
        if self.laplacian_n_permutaciones < 5:
            errores.append(
                f"laplacian_n_permutaciones={self.laplacian_n_permutaciones} debe ser >= 5 "
                "(con menos, el piso de ruido estimado por permutacion es demasiado ruidoso)."
            )
        if not 0 < self.alpha_ruido_laplaciano < 1:
            errores.append(f"alpha_ruido_laplaciano={self.alpha_ruido_laplaciano} debe estar en (0, 1).")
        suma_ns = self.peso_laplaciano + self.peso_dispersion
        if suma_ns <= 0:
            errores.append("peso_laplaciano + peso_dispersion debe ser mayor que cero.")
        elif abs(suma_ns - 1.0) > 1e-9:
            LOGGER.warning(
                "peso_laplaciano + peso_dispersion = %.4f != 1. Se renormalizaran a %.3f / %.3f.",
                suma_ns, self.peso_laplaciano / suma_ns, self.peso_dispersion / suma_ns,
            )

        if errores:
            raise ErrorConfiguracion(
                "Configuracion invalida:\n  - " + "\n  - ".join(errores)
            )

        # Normalizacion de pesos (post-validacion, para que sumen 1 exacto).
        total = self.peso_gini + self.peso_iv
        self.peso_gini /= total
        self.peso_iv /= total
        total_ns = self.peso_laplaciano + self.peso_dispersion
        self.peso_laplaciano /= total_ns
        self.peso_dispersion /= total_ns

        LOGGER.info("Configuracion validada correctamente.")
        LOGGER.info(
            "Columnas de rol -> target='%s' | id='%s' | tiempo='%s'",
            self.columna_target, self.columna_id, self.columna_tiempo,
        )


def _bloque_por_prefijo(nombre: str) -> str:
    """Asigna un bloque legible a cada parametro para la hoja de bitacora."""
    if nombre in {"usar_estandarizacion_puesto", "columnas_puesto_prioridad",
                  "columna_puesto_agrupado", "puesto_sin_dato"}:
        return "A2. Puestos de colaborador"
    if nombre.startswith(("umbral_ceros", "umbral_std", "umbral_cv", "umbral_dominancia",
                          "umbral_iqr", "umbral_alta_cardinalidad", "minimo_valores",
                          "usar_umbral")):
        return "B. Univariado"
    if nombre in ("usar_agrupacion_categorica_nombre", "umbral_cardinalidad_clustering",
                  "max_k_agrupacion_categorica"):
        return "B2. Agrupacion categorica por nombre"
    # OJO: se evalua ANTES que "C. Bivariado" porque "peso_laplaciano",
    # "peso_dispersion", "alpha_ruido_laplaciano" y "top_n_no_supervisado"
    # comparten prefijo con parametros de esa fase (peso_, alpha_ruido, top_n)
    # pero pertenecen a la rama SIN target, no a la de IV/Gini.
    if nombre in ("laplacian_k_vecinos", "laplacian_max_filas", "laplacian_n_permutaciones",
                  "alpha_ruido_laplaciano", "bonferroni_ruido_laplaciano", "peso_laplaciano",
                  "peso_dispersion", "top_n_no_supervisado"):
        return "C2. Bivariado (rama no supervisada)"
    if nombre.startswith(("n_bins", "min_prop", "max_categorias", "correccion_woe",
                          "peso_", "metodo_normalizacion", "umbral_iv", "umbral_gini",
                          "umbral_score", "excluir_sospecha", "top_n", "umbral_psi",
                          "excluir_por_inestabilidad", "usar_piso_ruido",
                          "alpha_ruido", "bonferroni_ruido")):
        return "C. Bivariado"
    if nombre.startswith(("umbral_correlacion", "metodo_correlacion", "umbral_vif", "excluir_por_vif")):
        return "D. Multivariado"
    if nombre.startswith(("motor_boruta", "boruta_")):
        return "E. Boruta"
    if nombre in {
        "usar_feature_engineering", "context_vars", "behavior_vars",
        "incluir_context_vars_en_seleccion", "kernel_type",
        "reference_mode", "min_group_size", "min_personal_history", "bandwidth_method",
        "bandwidth_grid", "temporal_windows", "epsilon", "kde_max_reference_rows",
        "preparar_modelos_anomalia", "meses_holdout_anomalia", "ruta_matriz_anomalias",
    }:
        return "G. Feature engineering de anomalias"
    return "F. Ejecucion"


# ---------------------------------------------------------------------------
# Carga desde YAML / dict
# ---------------------------------------------------------------------------
def cargar_config(
    ruta_yaml: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ConfigPipeline:
    """Construye la configuracion aplicando la precedencia documentada.

    Parameters
    ----------
    ruta_yaml
        YAML alternativo opcional. Completa sus claves con el config.yaml del
        proyecto. Un archivo ausente o invalido provoca ErrorConfiguracion.
    overrides
        Valores de mayor prioridad (tipicamente los flags del CLI). Las claves
        con valor ``None`` se ignoran para no pisar el YAML con "no informado".
    """
    datos: dict[str, Any] = {}

    if ruta_yaml is not None:
        ruta_yaml = Path(ruta_yaml)
        # La base se lee en ConfigPipeline; un YAML alternativo puede ser parcial.
        if ruta_yaml.resolve() != CONFIG_PREDETERMINADA.resolve():
            datos = _leer_yaml(ruta_yaml)

    if overrides:
        limpios = {k: v for k, v in overrides.items() if v is not None}
        if limpios:
            LOGGER.info("Overrides de linea de comandos: %s", limpios)
        datos.update(limpios)

    # Alias ingleses del contrato de FE solicitado. Se aceptan sin duplicar
    # estado dentro del dataclass: toda la aplicacion sigue leyendo un unico
    # nombre canonico.
    alias = {
        "id_col": "columna_id",
        "time_col": "columna_tiempo",
        "target_col": "columna_target",
        "random_state": "semilla",
    }
    for externo, canonico in alias.items():
        if externo not in datos:
            continue
        if canonico in datos:
            LOGGER.warning(
                "Se informaron '%s' y su alias '%s'; prevalece '%s'.",
                canonico, externo, canonico,
            )
        else:
            datos[canonico] = datos[externo]
        del datos[externo]

    validos = {f.name for f in fields(ConfigPipeline)}
    desconocidos = set(datos) - validos
    if desconocidos:
        LOGGER.warning("Parametros desconocidos ignorados: %s", sorted(desconocidos))

    cfg = ConfigPipeline(**{k: v for k, v in datos.items() if k in validos})
    cfg.validar()
    return cfg


def _leer_yaml(ruta: Path) -> dict[str, Any]:
    """Lee un YAML obligatorio sin reemplazar errores con defaults ocultos."""
    try:
        import yaml
    except ImportError as exc:
        raise ErrorConfiguracion(
            "PyYAML es necesario para leer la configuracion central. "
            "Instalelo con: py -m pip install PyYAML"
        ) from exc
    try:
        with ruta.open("r", encoding="utf-8") as archivo:
            crudo = yaml.safe_load(archivo)
        if not isinstance(crudo, dict):
            raise ErrorConfiguracion("La raiz del YAML debe ser un mapa de parametros.")
        datos = _aplanar_yaml(crudo)
    except (OSError, yaml.YAMLError, ErrorConfiguracion) as exc:
        raise ErrorConfiguracion(f"No se pudo leer '{ruta}': {exc}") from exc
    LOGGER.info("Configuracion leida desde %s (%d parametros).", ruta, len(datos))
    return datos


def _aplanar_yaml(crudo: dict[str, Any]) -> dict[str, Any]:
    """Aplana secciones del YAML y rechaza parametros repetidos entre bloques."""
    plano: dict[str, Any] = {}
    for clave, valor in crudo.items():
        valores = _aplanar_yaml(valor) if isinstance(valor, dict) else {clave: valor}
        repetidos = plano.keys() & valores.keys()
        if repetidos:
            raise ErrorConfiguracion(f"Parametros repetidos en el YAML: {sorted(repetidos)}")
        plano.update(valores)
    return plano
