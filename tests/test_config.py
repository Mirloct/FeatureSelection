"""Pruebas de la fuente central y del arranque del pipeline.

Data sources / inputs: config.yaml y YAML temporales de prueba.
Created: 2026-10-02
Last modified: 2026-10-04
Changelog:
- 2026-10-04: reemplaza los campos de ejemplo del feature engineering
  retirado por campos equivalentes que siguen existiendo en el esquema.
- 2026-10-02: verifica precedencia, independencia y opciones del arranque.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from featsel import config


def test_constructor_y_carga_comparten_fuente(monkeypatch, tmp_path):
    datos = yaml.safe_load(config.CONFIG_PREDETERMINADA.read_text(encoding="utf-8"))
    datos["entradas"]["columna_id"] = "cliente"
    datos["entradas"]["columnas_conservadas"] = ["saldo"]
    central = tmp_path / "config.yaml"
    central.write_text(yaml.safe_dump(datos), encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_PREDETERMINADA", central)
    monkeypatch.chdir(tmp_path.parent)
    assert config.ConfigPipeline().columna_id == "cliente"
    assert config.cargar_config().columnas_conservadas == ["saldo"]
    datos["entradas"]["columna_id"] = "persona"
    central.write_text(yaml.safe_dump(datos), encoding="utf-8")
    assert config.ConfigPipeline().columna_id == "persona"


def test_yaml_parcial_cli_y_aliases(tmp_path):
    alternativo = tmp_path / "parcial.yaml"
    alternativo.write_text("entradas:\n  columna_id: cliente\n  usar_boruta: false\n", encoding="utf-8")
    cfg = config.cargar_config(alternativo, {"columna_id": "persona", "semilla": None, "time_col": "mes"})
    assert cfg.columna_id == "persona"
    assert cfg.columna_tiempo == "mes"
    assert cfg.usar_boruta is False
    assert cfg.semilla == config.ConfigPipeline().semilla
    assert cfg.n_bins == config.ConfigPipeline().n_bins


def test_listas_no_se_comparten():
    primero = config.ConfigPipeline()
    segundo = config.ConfigPipeline()
    primero.columnas_conservadas.append("extra")
    primero.csv_encodings_fallback.append("cp850")
    assert "extra" not in segundo.columnas_conservadas
    assert "cp850" not in segundo.csv_encodings_fallback
    assert config.ConfigPipeline().columnas_conservadas == segundo.columnas_conservadas


def test_yaml_ausente_incompleto_y_repetido(monkeypatch, tmp_path):
    with pytest.raises(config.ErrorConfiguracion, match="No se pudo leer"):
        config.cargar_config(tmp_path / "ausente.yaml")
    repetido = tmp_path / "repetido.yaml"
    repetido.write_text("a:\n  semilla: 1\nb:\n  semilla: 2\n", encoding="utf-8")
    with pytest.raises(config.ErrorConfiguracion, match="repetidos"):
        config.cargar_config(repetido)
    incompleto = tmp_path / "incompleto.yaml"
    incompleto.write_text("semilla: 1\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_PREDETERMINADA", incompleto)
    with pytest.raises(config.ErrorConfiguracion, match="Faltan parametros"):
        config.ConfigPipeline()


@pytest.mark.parametrize("sin_autoinstall", [False, True])
def test_arranque_respeta_yaml(monkeypatch, tmp_path, sin_autoinstall):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    import run_pipeline
    from featsel import bootstrap, logging_utils, pipeline

    dataset = tmp_path / "panel.csv"
    dataset.write_text("id,mes\n1,202601\n", encoding="utf-8")
    ruta_log = str(tmp_path / "personal.log")
    alternativo = tmp_path / "personal.yaml"
    alternativo.write_text(yaml.safe_dump({
        "ruta_dataset": str(dataset), "usar_boruta": False,
        "autoinstalar_dependencias": not sin_autoinstall,
        "ruta_log": ruta_log, "nivel_log": "WARNING",
    }), encoding="utf-8")
    # Con el flag se comprueba que el CLI prevalece incluso sobre un YAML en true.
    if sin_autoinstall:
        datos = yaml.safe_load(alternativo.read_text(encoding="utf-8"))
        datos["autoinstalar_dependencias"] = True
        alternativo.write_text(yaml.safe_dump(datos), encoding="utf-8")
    arranque = Mock(return_value=object())
    configurar_log = Mock(return_value=object())
    ejecutar = Mock(return_value={"variables_seleccionadas": [], "boruta_meta": {}, "segundos": 0, "ruta_excel": "prueba.xlsx"})
    monkeypatch.setattr(bootstrap, "arrancar", arranque)
    monkeypatch.setattr(logging_utils, "configurar_logging", configurar_log)
    monkeypatch.setattr(pipeline, "ejecutar", ejecutar)
    argv = ["run_pipeline.py", "--config", str(alternativo)]
    if sin_autoinstall:
        argv.append("--sin-autoinstall")
    monkeypatch.setattr(sys, "argv", argv)
    assert run_pipeline.main() == 0
    configurar_log.assert_called_once_with(ruta_log, logging.WARNING)
    arranque.assert_called_once_with(usar_boruta=False, autoinstalar=not sin_autoinstall)
    assert ejecutar.call_args.args[0].ruta_dataset == str(dataset)
