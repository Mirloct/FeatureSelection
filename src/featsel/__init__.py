"""
featsel
=======

Pipeline modular, reproducible y auditable de seleccion de variables
(feature selection) para DATOS DE PANEL (entidad x tiempo).

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

__version__ = "1.0.0"
__all__ = [
    "bootstrap",
    "config",
    "logging_utils",
    "io_utils",
    "validaciones",
    "metricas",
    "fase0_diagnostico",
    "fase1_univariado",
    "fase2_bivariado",
    "fase3_multivariado",
    "fase4_boruta",
    "reporte_excel",
    "pipeline",
]
