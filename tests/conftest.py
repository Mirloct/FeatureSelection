"""
conftest.py
===========

Permite ejecutar `py -m pytest tests/` sin configurar PYTHONPATH a mano: el
paquete vive en `src/`, igual que resuelve `run_pipeline.py` en tiempo de
ejecucion (ver su propio `sys.path.insert`).

Data sources / inputs: ninguno (solo manipula sys.path).
Created: 2026-10-01
Last modified: 2026-10-01
Changelog:
- 2026-10-01: creado para que pytest encuentre `featsel` sin pasos manuales.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
