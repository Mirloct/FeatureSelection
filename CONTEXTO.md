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

El feature engineering temporal/KDE condicional y la estandarización de
puestos del colaborador que existieron en este proyecto (2026-10-01 a
2026-10-04) se retiraron el 2026-10-04: el usuario los implementó
directamente en su código base de entrada y ya no viven aquí (módulos,
tests, hojas de Excel y páginas de documentación específicas se eliminaron).
Lo que queda de ese periodo en `docs/auditoria_final.html` es un registro
histórico de la auditoría matemática/código/estructura, no una
funcionalidad vigente — ver la nota de actualización al inicio de esa página.

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
- La antigua lista que transportaba columnas ahora se llama
  `columnas_conservadas`; `columnas_excluidas` significa exclusión completa.

CSV reintenta los encodings configurados (`csv_encodings_fallback`)
únicamente ante `UnicodeDecodeError`; registra el fallback usado y mantiene
los demás errores de lectura sin tocar.

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
- El demo se genera automáticamente solo si falta el dataset, la carpeta está
  vacía y `generar_demo_si_falta` está activo. `--generar-demo` fuerza generación.

## Decisiones que deben preservarse

- Roles y columnas manuales quedan fuera de la selección estadística.
- Cálculo y reporte están separados; Boruta contrasta, no cambia la selección final.
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

Última suite ejecutada: `py -m pytest -q`, **38 pruebas aprobadas** el 2026-10-04
(tras retirar el feature engineering temporal/KDE y la estandarización de
puestos: tenía 115, tres archivos de test y un caso de fundamentos
matemáticos se eliminaron junto con ese código, ver sección final).
Cubre configuración/precedencia, opciones del arranque simulado, columnas
manuales, exportaciones CSV/Parquet y corridas completas del pipeline
(con/sin target, Excel y dataset final).
Los conteos y tiempos de demos históricos no son resultados de esta suite ni
una garantía del resultado de otra configuración.

No hay aún cobertura automatizada exhaustiva de WOE/IV/Gini/VIF/Laplacian y
clustering. Tampoco se hace evaluación out-of-time del modelo final.

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
| `src/featsel/reporte_excel.py` | Formato y exportación de bitácora |
| `tests/` | Pruebas de regresión |
| `docs/documentacion.html` | Fundamentos y detalle técnico |
| `docs/configuracion_centralizada.html` | Guía de configuración |
| `docs/exclusiones_manuales.html` | Guía de las dos listas |

## Validación matemática, de código y de estructura — 2026-10-04

Auditoría previa al retiro del feature engineering (registro histórico,
ver `docs/auditoria_final.html`):

1. Matemáticas: 9 referencias independientes para WOE/IV, AUC/Gini, PSI,
   eta/Cramér, VIF, energía laplaciana y densidad gaussiana (la referencia
   de KDE se retiró junto con ese módulo).
2. Código: 9 pruebas nuevas de decisiones y permutaciones (las 33 adversarias
   de KDE se retiraron junto con ese módulo).
3. Estructura: pruebas de pipeline con/sin target, Excel/CSV, neutralización
   de fórmulas y equivalencia del esquema/YAML (los casos de FE activo/apagado
   y TRAIN/HOLDOUT de la matriz IF/VAE se retiraron junto con ese módulo).

Correcciones de ese momento que siguen vigentes (no dependían de FE):
VIF singular por regresión (infinito para dependencia exacta), flag VIF alto
para infinito, p Monte Carlo (b+1)/(B+1), alpha sin recortes, vecinos sin
autoaristas y desempates exactos por score, luego IV/Gini. Spearman no
equivale a Somers D. La limpieza de comentarios de esa auditoría introdujo
una pérdida de campos del esquema; se restauraron y se agregó
`test_esquema_final_cubre_todo_el_yaml_central` para que no vuelva a pasar
inadvertido. Las reglas corregidas pueden cambiar la selección final
respecto a corridas anteriores a esa fecha.

Suite actual: 38 pruebas (ver "Verificación y límites" arriba).
