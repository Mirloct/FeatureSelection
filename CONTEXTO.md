# Contexto del proyecto

Actualizado: 2026-10-04. Guía de continuidad; el código y `config.yaml` determinan
el comportamiento vigente. Para uso consulte `README.md`; para fundamentos,
`docs/documentacion.html`.

## Propósito y flujo

Selección de variables para datos de panel: una fila por entidad y periodo.
Id y tiempo son obligatorios. La presencia del target decide el flujo:

- Con target: diagnóstico → univariado → agrupación categórica opcional →
  IV/Gini → asociación/VIF → Boruta opcional como contraste.
- Sin target: mismas etapas iniciales → Laplacian Score y dispersión robusta →
  asociación/VIF. Boruta no se ejecuta.
- Feature engineering opcional: temporales personales y KDE condicional antes
  del diagnóstico, común a ambas ramas. Las features generadas se evalúan como
  cualquier candidata.

## Configuración y columnas manuales

Todos los valores se editan en `config.yaml`. `src/featsel/config.py` conserva
el esquema, las propiedades derivadas y las validaciones, sin defaults duplicados.
`ConfigPipeline()` y `cargar_config()` leen el mismo YAML central.

Precedencia: YAML central del proyecto → YAML alternativo parcial (`--config`)
→ flags del CLI. Los overrides con valor `None` se ignoran.
La ruta central se resuelve desde el proyecto; las rutas de datos/salidas son
relativas al directorio de ejecución.

Las dos listas están juntas en `entradas`:

```yaml
columnas_conservadas: []  # no analizar; conservar al final del dataset
columnas_excluidas: []   # retirar antes del análisis; no exportar
```

- Conservadas: no son candidatas; mantienen sus valores y tipos cargados y van
  al final en el orden configurado, sin duplicados. Los nombres inexistentes se
  ignoran con advertencia.
- Excluidas: se retiran después de cargar, antes de tipificar y analizar.
- Una columna no puede estar en ambas listas ni ser id, tiempo o target.
- Una columna excluida no puede ser contexto/conducta de FE cuando esté activo.
- La antigua lista que transportaba columnas ahora se llama
  `columnas_conservadas`; `columnas_excluidas` significa exclusión completa.

## Puestos del colaborador

`src/featsel/puestos.py` prepara antes de tipificar, luego de aplicar exclusiones.
Prioridad central en `puestos_colaborador.columnas_puesto_prioridad`:
`despuestocolaborador` sobre `desposicioncolaborador`; la fuente sobrante se retira.
Si solo hay una, se usa esa. Sin fuente o con el interruptor apagado, no cambia
el DataFrame ni la configuración. Tolera mayúsculas, espacios y BOM en encabezados.

Crea `despuestocolaboradoragrupado`: reparación reversible cp1252/latin1→UTF-8,
normalización sin tildes, extracción de palabras y reglas léxicas deterministas.
Dos primeras palabras para familias/calificadores reconocidos; primera palabra
para desconocidos. GERENTE/GTE se unifican; GTEADJ y GERENTE ADJUNTO se unifican;
reconoce asistente, analista y otras familias habituales. Nulos, signos y números
sin palabras usan `SIN INFORMACION`; las letras irrecuperables no se adivinan.

Fuente elegida y agrupado se agregan a conservadas. El agrupado reemplaza las
fuentes en context_vars y se agrega como contexto KDE. Otros contextos se
mantienen; se retira el placeholder job_position solo si no existe. Se devuelve
una copia de cfg: no se cambia la configuración del llamador ni el YAML.
No activa FE: requiere el interruptor y conductas/contextos restantes válidos.
KDE conserva causalidad temporal y usa nulos si falta historia por grupo.
CSV reintenta los encodings configurados únicamente ante UnicodeDecodeError;
registra el fallback y mantiene los demás errores de lectura.

## Arranque y salidas

PyYAML es requisito previo: `py -m pip install PyYAML`.
Orden: configuración → logging → bootstrap → carga → fases → exportación.
El arranque respeta `ruta_log`, `nivel_log`, `usar_boruta` y
`autoinstalar_dependencias`. `--sin-autoinstall` desactiva la instalación.

- Excel: bitácora con diagnóstico, métricas, decisiones y parámetros.
- Dataset final CSV/Parquet: id → tiempo → target si existe → seleccionadas →
  conservadas. Las descartadas estadísticamente y las excluidas totalmente
  quedan fuera. Conserva las filas; no imputa para este archivo.
- Si no hay seleccionadas, se exporta cuando hay conservadas presentes. Sin
  ambas, se omite. `exportar_dataset_final: false` desactiva esta salida.
- Matriz IF/VAE: opcional con FE activo; utiliza seleccionadas, no las columnas
  transportadas manualmente. Preprocesamiento ajustado solo en TRAIN, corte
  temporal para HOLDOUT. El proyecto prepara la matriz, no entrena IF/VAE.
- El demo se genera automáticamente solo si falta el dataset, la carpeta está
  vacía y `generar_demo_si_falta` está activo. `--generar-demo` fuerza generación.

## Decisiones que deben preservarse

- Roles y columnas manuales quedan fuera de la selección estadística.
- Cálculo y reporte están separados; Boruta contrasta, no cambia la selección final.
- FE causal: una fila del periodo t usa únicamente periodos anteriores.
  Temporales con desplazamiento; KDE y elección de bandwidth sobre historia.
  Perfiles con historia insuficiente generan nulos, sin mezclar poblaciones.
- Imputación, codificación y escalado IF/VAE aprenden solo de TRAIN.
- IV utiliza piso de ruido; el contraste de Gini utiliza el valor bruto.
- Flags numéricos 0/1 con ambos valores presentes se retienen en fase 1 por
  excepción dicotómica; pueden descartarse en fases posteriores.
- Agrupación categórica por similitud de nombre corre una vez antes de separar
  ramas, sin usar target. Categorías raras se agrupan también en el binning.
- Asociación mixta: Spearman/Pearson para numéricas, Cramér para categóricas,
  eta para pares mixtos. VIF elimina solo si está configurado.
- Laplacian Score usa un grafo construido sin la variable evaluada para evitar
  circularidad. Las permutaciones determinan la evidencia frente al ruido.

## Verificación y límites

Última suite ejecutada: `py -m pytest -q`, **115 pruebas aprobadas** el 2026-10-04.
Cubre configuración/precedencia, opciones del arranque simulado, columnas manuales,
exportaciones CSV/Parquet, normalización de puestos, decoding CSV, KDE con el
agrupado y causalidad/preparación IF/VAE.
Los conteos y tiempos de demos históricos no son resultados de esta suite ni
una garantía del resultado de otra configuración. Las pruebas nuevas incluyen corridas completas de ambas ramas con un panel
sintético pequeño y FE activo; no equivalen a una corrida sobre datos reales.

No hay aún cobertura automatizada exhaustiva de WOE/IV/Gini/VIF/Laplacian y
clustering. Tampoco se entrena un modelo de anomalías ni se hace evaluación
out-of-time del modelo final.

## Retomar el trabajo

```powershell
py -m pytest -q
py run_pipeline.py
py run_pipeline.py --config otro.yaml --sin-autoinstall
```

Mapa de archivos:

| Archivo | Responsabilidad |
|---|---|
| `config.yaml` | Valores configurables |
| `src/featsel/config.py` | Esquema, carga y validación |
| `run_pipeline.py` | CLI y arranque |
| `src/featsel/pipeline.py` | Orquestación y dataset final |
| `src/featsel/io_utils.py` | Lectura y tipificación |
| `src/featsel/metricas.py` | Métricas y helpers estadísticos |
| `src/featsel/fase2_no_supervisado.py` | Relevancia sin target |
| `src/featsel/feature_engineering_anomalias.py` | FE causal y matriz IF/VAE |
| `src/featsel/reporte_excel.py` | Formato y exportación de bitácora |
| `src/featsel/puestos.py` | Preparación léxica y contexto de puestos |
| `tests/` | Pruebas de regresión |
| `docs/documentacion.html` | Fundamentos y detalle técnico |
| `docs/configuracion_centralizada.html` | Guía de configuración |
| `docs/exclusiones_manuales.html` | Guía de las dos listas |
| `docs/feature_engineering_anomalias.html` | Guía de FE |

Guía de puestos: `docs/puestos_colaborador.html`.

## Auditoría KDE — 2026-10-04

Se mantiene FE apagado con nombres generales por decisión del usuario.
Las columnas se editan en config.yaml: entradas para roles/listas manuales;
feature_engineering.context_vars para grupos y behavior_vars para conductas numéricas.

Corregido: pérdida de columnas auxiliares del usuario, colisiones al normalizar
nombres de features, mezcla de nulos con la categoría literal __MISSING__,
infinitos en conductas y parámetros KDE no finitos/ventanas fraccionarias.
Pruebas nuevas: 33 casos; incluyen los seis kernels, los tres modos de referencia,
columnas elegidas, futuro alterado e índices repetidos/filas desordenadas.
Informe: docs/feature_engineering_anomalias.html.

## Validación final en tres iteraciones — 2026-10-04

1. Matemáticas: 9 referencias independientes para WOE/IV, AUC/Gini, PSI,
   eta/Cramér, VIF, energía laplaciana, densidad gaussiana y rareza empírica.
2. Código: 9 pruebas nuevas de decisiones y permutaciones, más 33 adversarias KDE.
3. Estructura: 6 pruebas de pipeline con/sin target, FE activo/apagado, Excel/CSV,
   neutralización de fórmulas, TRAIN/HOLDOUT y equivalencia del esquema/YAML.

Correcciones: VIF singular por regresión (infinito para dependencia exacta),
flag VIF alto para infinito, p Monte Carlo (b+1)/(B+1), alpha sin recortes,
vecinos sin autoaristas y desempates exactos por score, luego IV/Gini.
Spearman no equivale a Somers D. La limpieza de comentarios introdujo una
pérdida de campos del esquema; se restauraron y se agregó una prueba de integridad.
Las reglas corregidas pueden cambiar la selección final respecto a corridas anteriores.
Suite final: 115 pruebas. FE continúa apagado. Informe: docs/auditoria_final.html.
