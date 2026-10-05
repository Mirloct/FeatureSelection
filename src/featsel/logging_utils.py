"""
logging_utils.py
================

Logging del proceso. Cumple dos funciones:

1. Mostrar el avance en consola y persistirlo en un archivo `.log`.
2. Capturar TODOS los mensajes en memoria para volcarlos despues a la hoja
   `08_Bitacora_Log` del Excel. Asi la bitacora es autocontenida: quien reciba
   el Excel puede reconstruir la corrida sin pedir el archivo de log.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

#: Formato unico para consola y archivo.
_FORMATO = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_FECHA = "%Y-%m-%d %H:%M:%S"


class ManejadorMemoria(logging.Handler):
    """Handler que acumula los registros en una lista de diccionarios.

    Se usa como fuente de la hoja de bitacora del Excel.
    """

    def __init__(self) -> None:
        super().__init__()
        self.registros: list[dict[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            self.registros.append(
                {
                    "timestamp": datetime.fromtimestamp(record.created).strftime(_FECHA),
                    "nivel": record.levelname,
                    "modulo": record.name,
                    "mensaje": record.getMessage(),
                }
            )
        except Exception:  # noqa: BLE001 - el logging nunca debe tumbar el proceso
            self.handleError(record)


def configurar_logging(
    ruta_log: str | Path | None = None,
    nivel: int = logging.INFO,
) -> ManejadorMemoria:
    """Configura el logger raiz `featsel` y devuelve el handler en memoria.

    Parameters
    ----------
    ruta_log
        Archivo donde persistir el log. Si es ``None`` solo se usa consola.
    nivel
        Nivel minimo (``logging.INFO`` por defecto; ``DEBUG`` para auditoria fina).
    """
    logger = logging.getLogger("featsel")
    logger.setLevel(nivel)
    logger.handlers.clear()          # evita duplicar salidas al reejecutar
    logger.propagate = False

    formatter = logging.Formatter(_FORMATO, datefmt=_FECHA)

    consola = logging.StreamHandler(stream=sys.stdout)
    consola.setFormatter(formatter)
    consola.setLevel(nivel)
    logger.addHandler(consola)

    if ruta_log is not None:
        ruta_log = Path(ruta_log)
        ruta_log.parent.mkdir(parents=True, exist_ok=True)
        archivo = logging.FileHandler(ruta_log, mode="w", encoding="utf-8")
        archivo.setFormatter(formatter)
        archivo.setLevel(logging.DEBUG)
        logger.addHandler(archivo)

    memoria = ManejadorMemoria()
    memoria.setLevel(nivel)
    logger.addHandler(memoria)

    return memoria


def obtener_logger(nombre: str) -> logging.Logger:
    """Devuelve un logger hijo del arbol `featsel`."""
    return logging.getLogger(f"featsel.{nombre}")


class BarraProgreso:
    """Barra de progreso del pipeline, con ``tqdm`` como motor principal.

    ``tqdm`` escribe a stderr y redibuja su propia linea con ``\\r``; el
    logging del pipeline escribe a stdout con lineas completas (ver
    ``configurar_logging``). Al ser dos streams distintos no se pisan entre
    si -- es exactamente el patron que tqdm esta disenado para soportar
    (barra + logging intercalados), por eso no hace falta coordinarlos a mano.

    Si ``tqdm`` no esta instalado (o `bootstrap` no pudo resolverlo), cae
    automaticamente a un contador de texto plano: una linea nueva por paso
    completado, sin depender de ninguna libreria externa. La interfaz
    publica (``BarraProgreso(pasos)`` + ``.avanzar()``) es identica en
    ambos casos, asi que el resto del pipeline no necesita saber cual de
    los dos esta activo.
    """

    def __init__(self, pasos: list[str], ancho: int = 30) -> None:
        self._pasos = pasos
        self._total = len(pasos) or 1
        self._ancho = ancho
        self._actual = 0
        self._tqdm = None
        try:
            from tqdm import tqdm

            self._tqdm = tqdm(
                total=self._total, unit="fase",
                bar_format="  {desc}: {bar} {n_fmt}/{total_fmt} ({percentage:3.0f}%)",
            )
        except ImportError:
            pass

    def avanzar(self, etiqueta: str | None = None) -> None:
        """Marca completado el siguiente paso y actualiza la barra."""
        self._actual = min(self._actual + 1, self._total)
        if etiqueta is None:
            etiqueta = self._pasos[self._actual - 1] if self._actual <= len(self._pasos) else ""

        if self._tqdm is not None:
            self._tqdm.set_description_str(etiqueta, refresh=False)
            self._tqdm.update(1)
            if self._actual >= self._total:
                self._tqdm.close()
            return

        pct = self._actual / self._total
        llenado = int(round(pct * self._ancho))
        barra = "#" * llenado + "-" * (self._ancho - llenado)
        print(f"  [{barra}] {self._actual}/{self._total} ({pct:.0%})  {etiqueta}", flush=True)
