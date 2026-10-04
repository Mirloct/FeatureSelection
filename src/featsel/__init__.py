"""
featsel
=======

Pipeline modular, reproducible y auditable de seleccion de variables
(feature selection) para DATOS DE PANEL (entidad x tiempo).

Data sources / inputs: panel configurado en ``config.yaml``.
Created: 2026-07-26
Last modified: 2026-10-04
Changelog:
- 2026-10-01: exporta el modulo opcional de FE temporal/KDE para anomalias.
- 2026-10-04: exporta ``puestos`` (preparacion lexica de puestos del colaborador).

Fases
-----
0. Diagnostico inicial del dataset.
1. Pruebas univariadas   (ceros+nulos, baja variacion).
2. Pruebas bivariadas    (Information Value, Gini, score compuesto).
3. Pruebas multivariadas (correlacion / asociacion, VIF, redundancia).
4. Prueba opcional       (Boruta / BorutaShap como contraste).

Toda la evidencia se exporta a un unico Excel multi-hoja de bitacora.

Principio de diseno
-------------------
La logica de CALCULO (metricas y fases) esta completamente separada de la
logica de EXPORTACION (`reporte_excel.py`). Las fases devuelven DataFrames
puros; el reporteador solo los formatea.
"""

__version__ = "1.2.0"
__all__ = [
    "bootstrap",
    "config",
    "logging_utils",
    "io_utils",
    "validaciones",
    "metricas",
    "fase0_diagnostico",
    "feature_engineering_anomalias",
    "puestos",
    "fase1_univariado",
    "fase1b_agrupacion_categorica",
    "fase2_bivariado",
    "fase2_no_supervisado",
    "fase3_multivariado",
    "fase4_boruta",
    "reporte_excel",
    "pipeline",
]
