# CONTEXTO DEL PROYECTO

Documento de continuidad: si el trabajo se retoma en otra sesión, esto basta
para entender el diseño sin releer todo el código ni el historial de chat.

---

## 1. Qué es este proyecto

Pipeline de selección de variables para **datos de panel** (entidad × tiempo),
con dos flujos según exista o no una columna target en el dataset:

- **Con target** (supervisado): Fase 0 diagnóstico → Fase 1 univariado →
  Fase 1B agrupación de categóricas por nombre → Fase 2 bivariado (IV/Gini) →
  Fase 3 multivariado (correlación/VIF) → Fase 4 Boruta (opcional).
- **Sin target** (no supervisado, fallback automático): Fase 0, 1 y 1B
  idénticas → Fase 2 alternativa (Laplacian Score + dispersión robusta) →
  Fase 3 idéntica → sin Fase 4. Orientado a alimentar Isolation Forest o un VAE.

La Fase 1B (`fase1b_agrupacion_categorica.py`) corre UNA SOLA VEZ, antes del
fork: no usa el target (agrupa por cómo se ESCRIBE el nombre de la categoría,
no por cómo se comporta frente al target), así que sirve a las dos ramas por
igual sin duplicar cómputo.

Cuál flujo corre se decide en un único punto —
`validaciones.target_disponible()`, justo tras cargar el dataset — sin que el
usuario tenga que configurar nada.

Todo el proceso queda documentado en un Excel de bitácora (19 hojas con
target, 16 sin target) más un CSV "listo para modelar" con solo las variables
seleccionadas.

Requisitos de diseño que se mantienen en todo el código:

- Toda exclusión lleva un flag y un motivo en texto; nada se elimina en silencio.
- Cálculo (fases, `metricas.py`) y exportación (`reporte_excel.py`) están
  separados: la capa de exportación no toma decisiones.
- Dependencias faltantes se instalan automáticamente (`bootstrap.py`) sin
  detener el flujo si una opcional falla.
- El proyecto trata el dataset como panel (entidad+tiempo), no como corte
  transversal: valida la llave compuesta, descompone varianza *within/between*.

---

## 2. Parámetros y nombres de columna: cero hardcodeo

Los nombres de `columna_target` / `columna_id` / `columna_tiempo` viven
**únicamente** en `config.yaml`, dentro del objeto `ConfigPipeline`
(`src/featsel/config.py`). Ningún módulo tiene un literal de nombre de
columna: todos leen `cfg.columna_target`, etc., o los helpers
`cfg.columnas_rol` / `cfg.columnas_no_candidatas` / `cfg.rol_de(columna)`.
Esto incluye el generador de datos de demostración y el reporteador Excel.

**Verificado:** el pipeline corre idéntico con las columnas renombradas
(`malo_90d`/`rut_cliente`/`mes_cierre`), pasando solo flags de CLI, sin tocar
código.

Parámetros de la corrida de referencia:

| Parámetro | Valor |
|---|---|
| `ruta_dataset` | `data/panel_sintetico.csv` (con target) / `data/panel_sin_target.csv` (sin target, mismo panel) |
| `columna_target` | `target` — binario 0/1 |
| `columna_id` | `id_entidad` |
| `columna_tiempo` | `periodo` |
| `usar_boruta` | `true` (se ignora automáticamente si no hay target) |
| `ruta_salida_excel` | `outputs/bitacora_feature_selection.xlsx` |

---

## 3. Estado: completo y verificado, ambos flujos

| Componente | Estado |
|---|---|
| Bootstrap de dependencias con autoinstalación | ✅ |
| Configuración centralizada + validación | ✅ |
| Validaciones de integridad del panel (target opcional, id+tiempo obligatorios) | ✅ |
| Fase 0 · Diagnóstico | ✅ |
| Fase 1 · Univariado | ✅ |
| Fase 2 · Bivariado IV/Gini (con target) | ✅ |
| Fase 2 · Laplacian Score (sin target) | ✅ |
| Fase 3 · Multivariado (asociación/VIF), común a ambos flujos | ✅ |
| Fase 4 · Boruta + implementación nativa (solo con target) | ✅ |
| Exportación Excel + dataset final listo para modelar | ✅ |
| Generador de panel sintético con patologías sembradas | ✅ |
| README, documentación HTML, este contexto | ✅ |

### Última corrida verificada

```
CON TARGET (data/panel_sintetico.csv, 6.923 filas × 31 columnas)
28 candidatas → 24 (fase 1, incluye 1 RETENIDA_DICOTOMICA) → 11 (fase 2) → 7 (fase 3) → Boruta: 6 confirmadas, 1 tentativa, 17 rechazadas
~30 s con Boruta, ~4 s sin él

SIN TARGET (data/panel_sin_target.csv, mismo panel sin la columna target)
28 candidatas → 24 (fase 1, incluye 1 RETENIDA_DICOTOMICA) → 18 (fase 2: Laplacian Score) → 14 (fase 3)
~8 s, sin fase 4
```

La diferencia de 23→24 en fase 1 frente a corridas previas es la excepción
dicotómica (§4): `var_casi_constante` (99.44% de un solo valor) ya no se
elimina por dominancia. En ambos flujos termina descartada más adelante por
falta de poder predictivo (fase 2), así que el resultado final no cambia —
solo cambia DÓNDE y POR QUÉ se descarta, que ahora es la razón correcta.

### Pruebas de regresión que deben seguir pasando

- `usar_boruta=true/false` → hoja `05_Boruta` presente/ausente según corresponda.
- Columnas de rol renombradas vía CLI → mismo resultado, sin tocar código.
- Llave `id+tiempo` duplicada → bloquea con mensaje explícito.
- Dataset sin columna target → activa el flujo no supervisado, no falla.
- Configuración inválida (ej. `umbral_correlacion=1.5`) → bloquea en validación.

---

## 4. Decisiones técnicas que no hay que deshacer

Cada una responde a un fallo real detectado durante el desarrollo. Si algo
aquí "se ve raro", es a propósito — revisar antes de "corregirlo".

**Piso de ruido estadístico (IV).** Un umbral fijo de IV (0.02) ignora que el
IV espurio de una variable aleatoria crece con el número de bins y decrece
con el tamaño muestral. El umbral efectivo es
`max(umbral_fijo, piso_de_ruido)`, con `piso_ruido_iv = c·χ²_{1−α}(k−1)`
donde `c = 1/n₁ + 1/n₀`. Parametrizable: `usar_piso_ruido`, `alpha_ruido`,
`bonferroni_ruido`.

**El contraste de Gini usa el bruto, no el de WOE.** `gini_woe` se ajusta con
los mismos datos que se evalúan (consume grados de libertad) y está sesgado
al alza; comparado contra un piso derivado del AUC no ajustado, rescataba
variables de ruido puro. Se usa `gini_bruto` para el contraste; categóricas
(sin Gini no ajustado posible) dependen solo del IV.

**`MIN_NIVELES_CATEGORICOS = 8` en el binning.** Sin esto, una categórica de
alta cardinalidad con categorías muy repartidas (ej. 180 niveles al ~0.5%
cada uno) caía entera bajo `min_prop_bin` y colapsaba a un solo bin → IV=0
por artefacto del binning, no por falta de señal.

**Boruta es contraste, nunca reemplazo.** Mide importancia condicional
(con las demás variables presentes); IV/Gini miden poder marginal. Corre
sobre los sobrevivientes de la fase 1 (no de la fase 3) para poder opinar
también sobre lo que las fases 2 y 3 descartaron. Nunca modifica la selección
final. En panel las observaciones no son independientes (misma entidad en
varios periodos), lo que puede optimizar la importancia del Random Forest —
se documenta como reserva, no se corrige.

**Boruta nativo como fallback.** BorutaPy depende de alias de numpy retirados
y no siempre instala limpio. Hay una implementación nativa (shadow features +
test binomial con Bonferroni) que solo necesita scikit-learn. Motor `auto`:
`borutapy → borutashap → nativo`.

**Matriz de asociación mixta (fase 3).** Un único umbral (0.90) aplica a
tipos mixtos normalizando tres medidas a [0,1]: `|Spearman|` (num-num),
V de Cramér con corrección de Bergsma (cat-cat), razón de correlación η
(num-cat). Spearman por defecto: robusto a outliers y a relaciones no
lineales-pero-monótonas, comunes en variables económicas.

**Excepción dicotómica en la fase 1 (flags 0/1).** Un flag binario con
incidencia baja (ej. 99% ceros) dispara los criterios de ceros+nulos y de
categoría dominante, pero ahí el desbalance ES la señal, no ruido: eliminarlo
sería descartarlo precisamente por ser informativo. Una columna con
`mínimo=0` y `máximo=1` (ambos valores presentes) se retiene siempre
(`decision_univariada = RETENIDA_DICOTOMICA`), con el motivo que la habría
eliminado igual escrito para que la excepción sea auditable. No aplica a
categóricas de dos niveles no numéricas (ej. "SI"/"NO" como texto): solo a
columnas ya codificadas en {0, 1}.

**Avisos de categóricas: dominancia en zona gris y cardinalidad alta.** Dos
situaciones que no ameritan eliminación automática pero sí condicionan la
codificación (one-hot, WOE, target encoding) rio abajo, y por eso se
reportan como aviso (no tocan `flg_seleccionada_univariada`):
`flg_dominancia_alta` (una categoría cubre entre `umbral_dominancia_aviso`
=90% y `umbral_dominancia`=99%: riesgo de sesgo hacia la clase dominante al
codificar) y `flg_alta_cardinalidad` (más de `umbral_alta_cardinalidad`=20
niveles distintos: el one-hot deja de ser práctico). El pipeline ya evita el
problema de fondo sin necesidad de eliminar nada: la rama supervisada agrupa
categorías raras/excedentes en `__OTROS__` antes del binning WOE
(`max_categorias`), y la rama sin target (además de Boruta) usa codificación
ordinal por frecuencia en vez de one-hot en `construir_matriz_numerica`. Ver
`docs/documentacion.html` §8.3-8.4 para la comparación completa contra target
encoding, hashing, embeddings y MCA, y por qué no se adoptaron en su lugar.

**Agrupación de categóricas de cardinalidad muy alta por similitud de nombre
(fase 1B).** Ni la dominancia (§ arriba) ni el agrupamiento por frecuencia de
la fase 2 (`max_categorias` → `__OTROS__`) resuelven bien el caso de una
categórica genuinamente dispersa (ninguna categoría domina) pero con +100
niveles: un one-hot o un binning directo diluye la señal en columnas casi
vacías. Se agrupan los NOMBRES (no el comportamiento frente al target) con
TF-IDF de 3-gramas de caracteres (Cerda, Varoquaux y Kegl, 2018,
arXiv:1806.00979) + K-Means, eligiendo k por silueta media (Rousseeuw, 1987)
en el rango `[2, max_k_agrupacion_categorica]`. El cluster con MAYOR
dispersión interna (distancia coseno media al centroide) se etiqueta
`__OTROS__` (misma constante que usa `binear_categorica`); los demás se
etiquetan con su miembro más frecuente en los datos reales. Corre una sola
vez, común a ambas ramas, justo después de la fase 1 y antes del fork.
Verificado: `cat_alta_cardinalidad` (180 niveles sintéticos) → 19 grupos
(k por silueta=0.195, reducción 89%); el IV posterior (0.018) confirma que
la reducción no inventa señal, solo evita 180 bins casi vacíos.
Parámetros: `usar_agrupacion_categorica_nombre`, `umbral_cardinalidad_clustering`
(100), `max_k_agrupacion_categorica` (30). Ver `docs/documentacion.html` §10b.

**Laplacian Score: el grafo debe ser *leave-one-out*.** La primera versión
construía un único grafo con todas las variables juntas y evaluaba cada una
(real y permutada) contra ese mismo grafo. Con 10 columnas de ruido puro sin
relación entre sí, **ninguna se excluía nunca**: circularidad — si la
variable participa en construir el grafo, el grafo "sabe" dónde están sus
valores y parece artificialmente suave sobre su propio grafo. Corrección: el
grafo que puntúa la columna *j* se construye **solo con las demás columnas**.
Verificado: 9/10 ruido correctamente descartado, 2/2 señales inyectadas
correctamente detectadas (p=0.000). `laplacian_n_permutaciones` subió de 20 a
200 (con 20, la resolución del p-valor era más gruesa que alpha=0.05).
Detalle completo con números en `docs/documentacion.html` §19.3.

---

## 5. Comportamientos correctos que podrían parecer errores

- **Boruta rechaza predictores legítimos** cuando hay una variable con fuga
  de información (`var_fuga`, IV=11.4): acapara la importancia del bosque y
  hunde a las demás bajo el umbral de las sombras. El resumen ejecutivo lo
  advierte automáticamente.
- **Variables al límite exacto de un umbral quedan excluidas** (ej. IV muy
  cercano a 0.02). Es la frontera de Siddiqi aplicada literalmente; si el
  negocio la quiere, se ajusta el umbral en `config.yaml`.
- **Entre dos variables casi idénticas, gana la de mayor `score_compuesto`**,
  aunque la diferencia sea marginal — es la regla especificada, no un empate
  mal resuelto.
- **El panel de demostración es desbalanceado a propósito** (se elimina ~4%
  de las filas) para ejercitar la detección de desbalance.
- **En el flujo sin target, 2-3 columnas de ruido puro pueden sobrevivir la
  fase 2** en una corrida puntual: a alpha=0.05 sin corrección de Bonferroni,
  con ~23 variables se esperan ~1 falso positivo solo por azar. Activar
  `bonferroni_ruido_laplaciano` lo corrige (subiendo también
  `laplacian_n_permutaciones` para tener resolución suficiente).

---

## 6. Extensiones pendientes (ninguna necesaria para lo pedido)

- **Validación out-of-time**: entrenar en los primeros periodos, evaluar en
  los últimos. Se recomienda en las conclusiones del Excel pero no se ejecuta.
- **Binning supervisado/monotónico** como alternativa a los cuantiles.
- **Clustering de variables** (ej. VarClusHi) en vez de eliminación greedy
  por pares en la fase 3.
- **Tests unitarios con pytest** sobre `metricas.py`.
- **Selección con efectos fijos**: transformar *within* antes de medir poder
  predictivo.
- **Concrete Autoencoders** (Abid, Balın y Zou, ICML 2019) como selección
  acoplada nativamente al VAE final, si se quiere ir más allá del filtro
  estadístico actual — evaluado y descartado por ahora por requerir entrenar
  una red dentro de la etapa de selección (ver `docs/documentacion.html` §19.6).

---

## 7. Cómo retomar

```powershell
cd c:\Users\Marco\Documents\Proyectos\FeatureSelection
py run_pipeline.py                                              # flujo con target
py run_pipeline.py --ruta-dataset data/panel_sin_target.csv `
                   --ruta-salida-excel outputs/bitacora_no_supervisada.xlsx   # flujo sin target
```

Puntos de entrada:

- Configuración y umbrales → [config.yaml](config.yaml)
- Orquestador (decide el flujo) → [src/featsel/pipeline.py](src/featsel/pipeline.py)
- Estadística supervisada (WOE, IV, Gini, VIF, PSI) → [src/featsel/metricas.py](src/featsel/metricas.py)
- Fase 1B, agrupación de categóricas por nombre (ambas ramas) → [src/featsel/fase1b_agrupacion_categorica.py](src/featsel/fase1b_agrupacion_categorica.py)
- Fase 2 no supervisada (Laplacian Score) → [src/featsel/fase2_no_supervisado.py](src/featsel/fase2_no_supervisado.py)
- Explicación completa con fuentes citadas → [docs/documentacion.html](docs/documentacion.html)
- Resumen orientado a uso → [README.md](README.md)
