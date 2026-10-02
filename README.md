# Feature Selection para Datos de Panel

<!--
Data sources / inputs: config.yaml y dataset declarado en entradas.ruta_dataset.
Created: 2026-07-26
Last modified: 2026-10-01
Changelog: 2026-10-01 - FE temporal/KDE opcional, matriz time-safe IF/VAE y
guía paso a paso para activar correctamente la rama.
-->

Pipeline modular, reproducible y auditable de selección de variables sobre datos
de panel (entidad × tiempo). Ejecuta validaciones **univariadas**, **bivariadas**,
**multivariadas** y una prueba **opcional** con **Boruta / BorutaShap**, deja
toda la evidencia en un único Excel de bitácora, y exporta además un
**dataset listo para modelar** con solo las variables seleccionadas.

Si el dataset **no tiene columna target**, el pipeline no falla: activa
automáticamente un **flujo no supervisado** (Laplacian Score + dispersión
robusta, sin Boruta) pensado para alimentar **Isolation Forest** o un
**autoencoder variacional (VAE)** — ver sección 9.

> **Estado: funcionando y verificado end-to-end**, en ambos flujos.
> Rama supervisada: 6.923 filas × 31 columnas → **7 variables seleccionadas**
> de 28 candidatas, ~30 s con Boruta (~4 s sin él).
> Rama no supervisada (mismo panel, sin target): 28 candidatas → **11
> seleccionadas**, ~10 s.

---

## 1. Arranque rápido

```powershell
py run_pipeline.py
```

### Activación correcta de la rama de feature engineering

El interruptor está apagado por defecto. **No basta con activar el flag** si
`context_vars` y `behavior_vars` todavía contienen los nombres de ejemplo:
primero deben reemplazarse por columnas que realmente existan en el dataset.

1. Configure las columnas de rol y la ruta del panel:

   ```yaml
   entradas:
     ruta_dataset: "data/mi_panel.csv"
     columna_id: "customer_id"
     columna_tiempo: "month"
     columna_target: "fraud_flag"  # puede no existir: activa el flujo sin target
   ```

2. Configure el perfil de comparación y las conductas mensuales. Las
   `behavior_vars` deben ser numéricas después de la tipificación:

   ```yaml
   feature_engineering:
     usar_feature_engineering: true
     context_vars: ["age_group", "job_position"]
     behavior_vars: ["credit_count", "credit_amount", "account_balance"]
     incluir_context_vars_en_seleccion: false
     kernel_type: "gaussian"
     reference_mode: "joint_and_marginal"
     min_group_size: 100
     min_personal_history: 6
     bandwidth_method: "time_safe_cv"
     epsilon: 1.0e-8
     meses_holdout_anomalia: 3
   ```

3. Ejecute el pipeline normalmente:

   ```powershell
   py run_pipeline.py --config config.yaml
   ```

   Como alternativa, deje `usar_feature_engineering: false` en el YAML y
   active únicamente esa corrida por CLI. Las listas de contexto y conducta
   se siguen leyendo del YAML:

   ```powershell
   py run_pipeline.py --config config.yaml --usar-feature-engineering true
   ```

Antes de generar variables, el pipeline exige una llave `id × month` única,
columnas contextuales existentes, conductas numéricas y más periodos que
`meses_holdout_anomalia`. Si algo no coincide, se detiene con un mensaje
explícito; no sustituye silenciosamente columnas ni poblaciones de referencia.

Al activarse, cada conducta produce lags/cambios/ventanas personales y rareza
por KDE frente a perfiles contextuales conjuntos y/o marginales. Solo se usa
historia anterior al mes evaluado. Todas las columnas generadas ingresan luego
a las fases normales de depuración y pueden ser descartadas si no aportan.

La misma corrida crea:

- hojas `01e`–`01h` en la bitácora con catálogo, estabilidad mensual, ablación
  por familia y parámetros de preprocesamiento;
- `<bitacora>_matriz_anomalias.csv`, numérica y sin nulos, con `TRAIN/HOLDOUT`
  temporal. Imputación, frequency encoding y escalado robusto se aprenden solo
  sobre `TRAIN`, por lo que sirve como contrato común para Isolation Forest/VAE.

Para compatibilidad con configuraciones en inglés también se aceptan los alias
`id_col`, `time_col`, `target_col` y `random_state`; internamente se normalizan
a la única configuración canónica del proyecto.

Eso es todo. En la primera ejecución el proyecto:

1. verifica e **instala automáticamente** las dependencias que falten,
2. genera un **panel sintético de demostración** si no encuentra el dataset,
3. ejecuta las cuatro fases,
4. escribe `outputs/bitacora_feature_selection.xlsx`.

Contra tus propios datos:

```powershell
py run_pipeline.py `
    --ruta-dataset       datos/mi_panel.csv `
    --columna-target     malo_90d `
    --columna-id         rut_cliente `
    --columna-tiempo     mes_cierre `
    --usar-boruta        true `
    --ruta-salida-excel  outputs/bitacora.xlsx
```

Suite de tests (cubre hoy el feature engineering opcional: interruptor
apagado, no-look-ahead, contrato de la matriz IF/VAE):

```powershell
py -m pytest tests/ -v
```

---

## 2. Los nombres de columna no están escritos en el código

Las columnas `target`, `id` y `tiempo` se declaran **en un solo lugar** y se
propagan a todo el proyecto vía el objeto `ConfigPipeline`. Ningún módulo
contiene literales como `"target"` o `"periodo"`: todos leen
`cfg.columna_target`, `cfg.columna_id`, `cfg.columna_tiempo`.

Cambiar estas tres líneas de [config.yaml](config.yaml) basta para que se adapten
el diagnóstico, las cuatro fases, la descomposición de varianza del panel, el
Excel **y hasta el generador de datos de demostración**:

```yaml
entradas:
  columna_target: "malo_90d"
  columna_id:     "rut_cliente"
  columna_tiempo: "mes_cierre"
```

Verificado con una corrida completa sobre el mismo dataset renombrado: no hay
que tocar una sola línea de código.

**Precedencia:** valores por defecto del código < `config.yaml` < flags del CLI.

---

## 3. Estructura del proyecto

```
FeatureSelection/
├── run_pipeline.py            ← PUNTO DE ENTRADA (CLI + arranque)
├── generar_datos_demo.py      ← panel sintético con patologías sembradas
├── config.yaml                ← FUENTE ÚNICA DE VERDAD de la configuración
├── requirements.txt
├── README.md                  ← este archivo
├── CONTEXTO.md                ← contexto del proyecto y estado de avance
│
├── src/featsel/
│   ├── bootstrap.py           ← revisa → instala → confirma dependencias
│   ├── config.py              ← dataclass de parámetros + validación
│   ├── logging_utils.py       ← logging a consola, archivo y memoria
│   ├── io_utils.py            ← carga y tipificación de datos
│   ├── validaciones.py        ← integridad del panel (llave id+tiempo, balance)
│   ├── metricas.py            ← WOE, IV, Gini, Cramér, VIF, PSI, piso de ruido, clustering de nombres
│   ├── feature_engineering_anomalias.py ← FE opcional · temporales causales + KDE condicional
│   ├── fase0_diagnostico.py   ← FASE 0 · diagnóstico inicial
│   ├── fase1_univariado.py    ← FASE 1 · ceros+nulos y baja variación
│   ├── fase1b_agrupacion_categorica.py ← FASE 1B · agrupa categóricas de cardinalidad muy alta por nombre
│   ├── fase2_bivariado.py     ← FASE 2 (con target) · IV, Gini y score compuesto
│   ├── fase2_no_supervisado.py← FASE 2 (sin target) · Laplacian Score + dispersión
│   ├── fase3_multivariado.py  ← FASE 3 · correlación, redundancia y VIF
│   ├── fase4_boruta.py        ← FASE 4 · Boruta / BorutaShap (opcional, solo con target)
│   ├── reporte_excel.py       ← EXPORTACIÓN (sin lógica de decisión)
│   └── pipeline.py            ← orquestador (bifurca según haya target o no)
│
├── tests/
│   └── test_feature_engineering_anomalias.py ← causalidad + contrato de la matriz IF/VAE
│
├── data/
│   ├── panel_sintetico.csv    ← generado automáticamente (con target)
│   └── panel_sin_target.csv   ← ejemplo del flujo no supervisado
├── docs/
│   ├── documentacion.html     ← documentación técnica y estadística completa
│   └── feature_engineering_anomalias.html ← guía ilustrada del FE opcional
└── outputs/
    ├── bitacora_feature_selection.xlsx
    ├── bitacora_feature_selection_dataset_final.csv    ← id+tiempo+target + seleccionadas
    ├── bitacora_no_supervisada.xlsx                    ← ejemplo del flujo sin target
    ├── *_matriz_anomalias.csv                          ← solo si el FE opcional está activo
    └── featsel.log
```

**Separación estricta:** las fases devuelven DataFrames puros;
`reporte_excel.py` solo formatea. Ninguna decisión de selección se toma en la
capa de exportación — si estuviera ahí, la trazabilidad se rompería.

---

## 4. Las fases (y su alternativa sin target)

| Fase | Qué hace **con** target | Qué hace **sin** target (fallback, §9) |
|---|---|---|
| **FE** *(opcional, apagado por defecto)* | Temporales causales (lag/diff/media/std/z) + rareza por KDE condicional, antes de todo lo demás. Ver «Activación correcta de la rama de feature engineering» en §1 | Idéntico — no usa el target |
| **0. Diagnóstico** | Perfil de cada columna: nulos, ceros, percentiles 25/50/75/90/99, únicos, varianza *within*/*between*, ICC | Idéntico — no usa el target |
| **1. Univariado** | Exceso de ceros+nulos · baja variación (`ceros+nulos ≥ 95%`, std≈0, CV≈0, dominancia ≥99%) | Idéntico — no usa el target |
| **1B. Agrupación categórica** | Categóricas de cardinalidad muy alta (>100 niveles, sin categoría dominante) se agrupan por similitud de nombre (TF-IDF + K-Means) | Idéntico — no usa el target |
| **2. Bivariado** | Information Value + Gini → `score_compuesto`. Excluye si IV **y** Gini bajo umbral | **Laplacian Score** + dispersión robusta → `score_compuesto`. Excluye si no supera su piso de ruido por permutación |
| **3. Multivariado** | Asociación por pares (`\|asoc.\| > 0.90`) + VIF; gana la de mayor `score_compuesto` | Idéntico — el `score_compuesto` de cualquiera de las dos fases 2 es intercambiable aquí |
| **4. Boruta** *(opcional)* | Importancia condicional vs. *shadow features* | **No se ejecuta**: exige un target contra el que entrenar el contraste |

---

## 5. Criterios: por qué esos umbrales

### Ceros + nulos ≥ 95%
Una variable constante en el 95% de las filas aporta señal en ≤5% de la muestra.
En panel esa fracción suele concentrarse en unas pocas entidades, de modo que el
modelo terminaría aprendiendo *el identificador* y no el fenómeno. Se ofrece un
umbral alterno más conservador del **90%** (`usar_umbral_alterno: true`).

### Baja variación — cinco pruebas, no una
Ninguna prueba basta sola, por eso se aplican en conjunto:

| Subcriterio | Detecta |
|---|---|
| `varianza = 0` / `std ≤ 1e-8` | Constante en escala **absoluta** |
| `CV = std/\|media\| ≤ 1e-4` | Constante en escala **relativa** (std de 0.01 es nada si la media es 10⁶) |
| dominancia ≥ 99% | Casi constante — funciona en numéricas **y** categóricas |
| `IQR ≈ 0` **y** `p25 = p99` | La distribución colapsa en un punto aunque tenga colas |
| `n_únicos < 2` | Constante pura |

### Correlación > 0.90
Umbral estándar para *redundancia severa*: por debajo de 0.90 dos variables
suelen conservar información propia. Se usa **Spearman** por defecto (invariante
a transformaciones monótonas y robusto a outliers, habituales en variables
económicas) y una **matriz de asociación mixta** en [0,1] para que un único
umbral aplique a todos los tipos:

- numérica–numérica → `|Spearman|`
- categórica–categórica → **V de Cramér** con corrección de Bergsma
- numérica–categórica → **razón de correlación η**

En cada par redundante **gana la de mayor `score_compuesto`**; el desempate
exacto lo resuelve el IV.

### Combinación de Gini e IV
Miden cosas distintas y son complementarias:

- El **Gini** (`2·AUC−1`) mide capacidad de **ordenamiento** global, pero penaliza
  las relaciones no monótonas.
- El **IV** mide información acumulada **bin a bin**, capta relaciones en U o por
  tramos, pero se infla con bins de poca masa.

No son comparables en crudo (el Gini está acotado en [0,1], el IV no tiene cota),
así que se **normalizan** antes de ponderar. Por defecto **50/50** para no
privilegiar de antemano ninguna visión; los pesos son parametrizables.

> El caso `var_no_monotona` del demo lo ilustra: **Gini crudo = 0.005** (invisible)
> frente a **IV = 0.445** (fuerte). Con una sola métrica se habría perdido.

### Piso de ruido (mejora sobre los umbrales fijos)
Un umbral fijo de IV ignora que **el IV espurio crece con el número de bins y
decrece con el tamaño de la muestra**. Una variable puramente aleatoria obtiene
siempre IV > 0. Bajo la hipótesis nula, con `c = 1/n₁ + 1/n₀`:

```
E[IV | H₀] = (k−1)·c        y        IV/c ~ χ²(k−1)
piso = c · χ²_{1−α}(k−1)
```

El umbral efectivo es `max(umbral_fijo, piso_de_ruido)`. Con la muestra del demo
(850 eventos, 6.073 no eventos, 10 bins) el piso es **IV ≤ 0.0291**: sin él, las
cinco variables `var_ruido_*` sobrevivían la fase 2. Con él, se eliminan las cinco.

El contraste del Gini usa **`gini_bruto`** y no `gini_woe`, porque este último se
calcula sobre un mapeo ajustado con los mismos datos (consume *k−1* grados de
libertad) y está sesgado al alza.

### Boruta / BorutaShap: contraste, no reemplazo
Boruta mide importancia **condicional** (en presencia de las demás variables);
IV y Gini miden poder **marginal**. Las discrepancias son informativas:

- IV alto + Boruta rechaza → la variable está **subsumida** por otra.
- IV bajo + Boruta confirma → aporta solo en **interacción**.

Reservas documentadas: en panel las observaciones **no son independientes** (una
entidad aparece en varios periodos), lo que optimiza la importancia estimada por
el Random Forest, cuya importancia además está sesgada hacia variables de alta
cardinalidad. Por eso Boruta **informa y matiza, pero no decide**.

---

## 6. Métricas específicas de panel

El dataset **no se trata como transversal**. Se añade:

- **Validación de la llave compuesta `id + tiempo`** — bloqueante: un panel exige
  una observación por entidad y periodo. Distingue duplicado exacto (basura de
  carga) de duplicado de llave con contenido distinto (granularidad mal declarada).
- **Descomposición de varianza *within* / *between* e ICC** — una variable con
  varianza total alta pero *within* nula es un atributo fijo de la entidad; un
  modelo de efectos fijos la absorbería por completo.
- **Estabilidad temporal**: IV calculado **por periodo** con bins globales fijos
  (re-binear por periodo haría incomparables los tramos), su coeficiente de
  variación, y el **PSI** contra el periodo base.
- **Balance del panel**: si está desbalanceado, las métricas agrupadas dan más
  peso a las entidades con más periodos observados. Se documenta explícitamente.
- **Submuestreo estratificado por periodo** en Boruta, para no sobrerrepresentar
  unos periodos frente a otros.

---

## 7. El Excel de bitácora (20 hojas con target · 17 sin target; +4 con feature engineering activo)

| Hoja | Contenido | ¿Solo con target? |
|---|---|---|
| `00_Resumen` | Resumen ejecutivo: entradas, embudo, criterios, calidad, conclusiones | No |
| `00_Embudo` | Cuántas variables entran / se eliminan / sobreviven por fase | No |
| `01_Diagnostico_Inicial` | Perfil completo de **cada** columna | No |
| `01b_Diagnostico_General` | Métricas de cabecera del dataset y del panel | No |
| `01c_Validacion_Panel` | Llave id+tiempo, balance, target | No |
| `01d_Target_por_Periodo` | Distribución temporal del target | **Sí** |
| `01e`–`01h` | Feature engineering: catálogo, estabilidad, ablación, preparación IF/VAE | Solo si `usar_feature_engineering=true` |
| `02_Univariado` | **Todas** las columnas originales con sus flags y motivos | No |
| `02b_Agrupacion_Categoricas` | Categóricas de cardinalidad muy alta agrupadas por nombre (fase 1B) | Solo si hubo algo que agrupar |
| `03_Bivariado` | IV, Gini, score compuesto, pisos de ruido, PSI | **Sí** |
| `03_Relevancia_NoSuperv` | Laplacian Score, piso de ruido, score compuesto | Solo **sin** target |
| `04_Multivariado` | Asociación máxima, VIF, redundancias, selección final | No |
| `05_Boruta` | Solo si `usar_boruta = true` **y** hay target | **Sí** |
| `06_Seleccion_Final` | Lista final para modelado | No |
| `06b_Descartadas` | Cada variable descartada, **en qué fase y por qué** | No |
| `07_Parametros` | Configuración exacta de la corrida | No |
| `07b_Dependencias` | Librerías verificadas / instaladas automáticamente | No |
| `08_Bitacora_Log` | Log cronológico completo | No |
| `09_Diccionario` | Significado de cada columna de cada hoja | No |
| `A1`–`A2` | Matriz de asociación, pares redundantes | No |
| `A3_Tablas_WOE` | Tablas WOE por variable | **Sí** |

Formato: semáforo condicional en los flags (verde = retenida, rojo = eliminada,
ámbar = advertencia), filtros automáticos, paneles congelados y formato numérico
por tipo de columna.

**Ninguna variable se elimina sin dejar rastro:** cada flag va acompañado de una
columna de texto con la razón exacta y los valores que la motivaron.

---

## 8. Dataset final listo para modelar

Además del Excel de bitácora, el pipeline exporta **un dataset** con:

```
id_entidad | periodo | target | <solo las variables que superaron las 3 fases>
```

Es decir: las columnas de rol (`columna_id`, `columna_tiempo`, `columna_target`,
en ese orden) más **únicamente** las variables que sobrevivieron las fases 1, 2
y 3. Ninguna variable descartada aparece en este archivo — esa trazabilidad ya
vive en `06b_Descartadas` de la bitácora; mezclarla aquí contaminaría el insumo
directo del modelo.

En la corrida de referencia: 6.923 filas × **11 columnas** (3 de rol + 8
seleccionadas), guardado como
`outputs/bitacora_feature_selection_dataset_final.csv`.

### Configuración

```yaml
entradas:
  exportar_dataset_final: true      # false -> no se genera este archivo
  ruta_dataset_final: ""            # "" -> se deriva junto a ruta_salida_excel
  formato_dataset_final: "csv"      # "csv" | "parquet"
```

Si `ruta_dataset_final` se deja vacío, la ruta se **deriva automáticamente**
junto al Excel de bitácora (`<mismo_nombre>_dataset_final.<formato>`), para que
ambos artefactos de una misma corrida queden agrupados y sea evidente a qué
bitácora corresponde cada dataset. Si ninguna variable supera las tres fases,
no se genera el archivo (quedaría solo con las columnas de rol) y se registra
la advertencia en el log.

---

## 9. Fallback no supervisado: cuando no hay `columna_target`

Al cargar el dataset, el pipeline verifica si `columna_target` existe. **Si no
existe**, no falla: activa automáticamente un flujo alternativo de selección
de variables para modelos **no supervisados** (pensado para **Isolation
Forest** y **autoencoders variacionales — VAE**), con el mismo Excel de
bitácora, las mismas estadísticas y el mismo dataset final que el flujo normal.

```
Fase 0 (idéntica) → Fase 1 (idéntica) → Fase 2 ALTERNATIVA → Fase 3 (idéntica)
                                          Laplacian Score           Sin Fase 4:
                                          + dispersión robusta    Boruta exige
                                          (sin target)            un target
```

La fase 2 reemplaza IV/Gini —que exigen etiqueta— por el **Laplacian Score**
(He, Cai y Niyogi, NeurIPS 2005), contrastado contra un **piso de ruido por
permutación** construido con un grafo *leave-one-out* (cada variable se evalúa
contra un grafo armado con las demás, nunca consigo misma: una primera versión
sin esta corrección no lograba excluir ni siquiera columnas de ruido puro
generadas a propósito — el hallazgo y la corrección están documentados con
números reales en `docs/documentacion.html` §19.3). Boruta no corre en este
modo: no hay target contra el que entrenar su Random Forest de contraste.

En la carpeta `data/` de este proyecto hay un ejemplo real: `panel_sin_target.csv`
es el mismo panel sintético sin la columna `target`, con su bitácora generada
en `outputs/bitacora_no_supervisada.xlsx` (28 candidatas → 23 → 18 → **14
seleccionadas**, ~8 segundos).

Ver `docs/documentacion.html` §19 para la justificación teórica completa
(por qué Laplacian Score y no otra técnica, por qué la redundancia importa más
sin IV/Gini para Isolation Forest y VAE específicamente, y las alternativas de
la literatura consideradas y descartadas), con las fuentes citadas.

---

## 10. Instalación automática de dependencias

`src/featsel/bootstrap.py` usa **solo la librería estándar** (no puede importar
pandas: pandas podría ser justo lo que falta). Por cada dependencia:

1. verifica si el módulo es importable,
2. si no, ejecuta `python -m pip install --upgrade <paquete>` (sin fijar versión,
   para obtener la última estable),
3. **invalida las cachés del importador** y reintenta la importación en el
   proceso actual — sin este paso Python seguiría creyendo que no existe,
4. registra el resultado en el log y en la hoja `07b_Dependencias`.

Degradación controlada:

- Falla una dependencia **crítica** (pandas, numpy, scipy, sklearn, openpyxl) →
  aborta con mensaje claro.
- Falla una **opcional** → se registra y **el flujo continúa**. Si BorutaPy y
  BorutaShap no están disponibles, se usa la **implementación nativa de Boruta**
  incluida en el proyecto, que solo necesita scikit-learn.

Las dependencias de Boruta solo se resuelven si `usar_boruta = true`.
Con `--sin-autoinstall` se verifica sin tocar el entorno (entornos sellados).

---

## 11. Verificación: el demo tiene patologías sembradas

El panel sintético incluye a propósito una variable por cada criterio, lo que
permite comprobar que cada regla hace lo que promete. Resultado de la corrida:

| Variable sembrada | Debe detectarla | Resultado |
|---|---|---|
| `var_constante` | Fase 1.2 · varianza cero | ✅ eliminada |
| `var_casi_constante` (dominancia 99.4%) | Excepción dicotómica (Fase 1) + Fase 2 · sin poder predictivo | ✅ `RETENIDA_DICOTOMICA` en fase 1 (es un flag 0/1, la dominancia no basta para eliminarla), eliminada en fase 2 por IV/Gini bajos — la razón correcta |
| `var_cv_bajo` | Fase 1.2 · CV despreciable | ✅ eliminada |
| `var_muchos_ceros` (97%) | Fase 1.1 · exceso de ceros | ✅ eliminada |
| `var_muchos_nulos` (96%) | Fase 1.1 · exceso de nulos | ✅ eliminada |
| `var_ruido_1..5` | Fase 2 · piso de ruido | ✅ las 5 eliminadas |
| `var_fija_entidad` | Fase 2 + ICC≈1 | ✅ eliminada y marcada como invariante |
| `cat_alta_cardinalidad` (180 niveles) | Fase 1B · clustering por nombre (TF-IDF+K-Means) + Fase 2 · `__OTROS__` | ✅ Fase 1B: 180→19 grupos (k por silueta=0.195); Fase 2: 18 bins sobre esos 19 grupos, eliminada por IV (0.018 < 0.02) |
| `var_texto_numerico` | Tipificación + fase 3 | ✅ convertida y detectada (ρ = 1.000) |
| `var_clon_score` | Fase 3 · redundancia (ρ = 0.9998) | ✅ par resuelto por score |
| `var_no_monotona` | Fase 2 · IV capta lo que el Gini pierde | ✅ retenida (IV 0.445 / Gini crudo 0.005) |
| `var_fuga` | Fase 2 · `flg_sospecha_fuga` | ✅ marcada (IV 11.4) |

### Un hallazgo que el propio informe explica

Boruta rechazó predictores **legítimos** (`var_n_atrasos_12m`, `var_monto_deuda`).
No es un error: `var_fuga` acapara casi toda la importancia del bosque y hunde a
las demás por debajo del umbral de las *shadow features*. El resumen ejecutivo
emite automáticamente esa advertencia: **hay que eliminar la fuga y volver a
ejecutar la fase 4 antes de interpretar sus rechazos.**

### Casos límite conocidos (comportamiento correcto, no defectos)

- `cat_sector` es un predictor real pero queda excluida con **IV = 0.019872**
  frente al umbral fijo **0.02**. Está en el límite exacto de la frontera clásica
  de Siddiqi. El motivo queda escrito en `06b_Descartadas`; si el negocio la
  considera relevante, basta con bajar `umbral_iv_minimo`.
- `var_clon_score` desplazó a `var_score_riesgo` (la original) porque midió un
  `score_compuesto` marginalmente mayor. Es la regla especificada aplicada al pie
  de la letra; ambas quedan en la bitácora con su score.

---

## 12. Reproducibilidad

- `semilla` fija Boruta, el submuestreo y el generador de datos.
- La hoja `07_Parametros` guarda la configuración exacta de la corrida.
- La hoja `07b_Dependencias` guarda las versiones de cada librería.
- La hoja `08_Bitacora_Log` guarda el log completo.

Una misma semilla + un mismo dataset + un mismo `config.yaml` producen la misma
selección.

---

## 13. Requisitos

Python ≥ 3.10. En este equipo se verificó con **Python 3.13.1**, pandas 2.3.1,
numpy 2.2.6, scipy 1.16.3, scikit-learn 1.7.2, openpyxl 3.1.5, Boruta 0.4.3.

> En Windows, si `python` no está en el PATH, use el lanzador: `py run_pipeline.py`.

---

## 14. Documentación completa

[`docs/documentacion.html`](docs/documentacion.html) — documentación técnica y
estadística: definiciones formales, derivaciones (WOE, IV, Gini, V de Cramér,
VIF, PSI, piso de ruido, clustering de nombres por TF-IDF+K-Means, KDE
condicional para anomalías), el porqué de cada decisión de diseño, escalas de
interpretación y limitaciones conocidas. Ábrala en el navegador.

[`docs/feature_engineering_anomalias.html`](docs/feature_engineering_anomalias.html)
— guía ilustrada de una sola página (con diagrama de flujo) específica del
módulo de feature engineering temporal/KDE opcional; el detalle estadístico
completo vive en `documentacion.html` §7b.
