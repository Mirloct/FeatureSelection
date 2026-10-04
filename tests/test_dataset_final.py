"""Contrato de las columnas excluidas manualmente en el dataset final.

Data sources / inputs: DataFrames sinteticos y archivos CSV/Parquet temporales.
Created: 2026-10-02
Last modified: 2026-10-02
Changelog:
- 2026-10-02: comprueba orden, valores y exportacion sin candidatas seleccionadas.
"""

import pandas as pd
import pytest

from featsel.config import ConfigPipeline
from featsel.io_utils import tipificar_dataset
from featsel.pipeline import _exportar_dataset_final


@pytest.mark.parametrize("supervisado", [True, False])
@pytest.mark.parametrize("formato", ["csv", "parquet"])
@pytest.mark.parametrize("seleccionadas", [["senal"], []])
def test_manuales_viajan_al_final(tmp_path, supervisado, formato, seleccionadas):
    if formato == "parquet":
        pytest.importorskip("pyarrow")
    df = pd.DataFrame({
        "auditoria": ["0012", "0034"], "id": [1, 2], "mes": [202601, 202601],
        "target": [0, 1], "senal": [2.5, 3.5], "descartada": [0, 0],
        "nota": ["uno", None],
    })
    if not supervisado:
        df = df.drop(columns="target")
    cfg = ConfigPipeline(
        columna_id="id", columna_tiempo="mes", columna_target="target",
        columnas_conservadas=["nota", "auditoria", "nota", "ausente"],
        ruta_dataset_final=str(tmp_path / f"final.{formato}"),
        formato_dataset_final=formato, exportar_dataset_final=True,
    )
    normalizado, _ = tipificar_dataset(df, cfg)
    pd.testing.assert_series_equal(normalizado["auditoria"], df["auditoria"])
    pd.testing.assert_series_equal(normalizado["nota"], df["nota"])
    ruta = _exportar_dataset_final(normalizado, cfg, seleccionadas, supervisado)
    resultado = (pd.read_parquet(ruta) if formato == "parquet"
                 else pd.read_csv(ruta, dtype={"auditoria": str}))
    esperado = ["id", "mes"] + (["target"] if supervisado else []) + seleccionadas + ["nota", "auditoria"]
    assert resultado.columns.tolist() == esperado
    assert resultado["auditoria"].tolist() == ["0012", "0034"]
    assert resultado["nota"].iloc[0] == "uno"
    assert pd.isna(resultado["nota"].iloc[1])
    assert cfg.rol_de("auditoria") == "CONSERVADA_MANUAL"
    assert "auditoria" in cfg.columnas_no_candidatas


def test_exportacion_desactivada_y_sin_columnas_utiles(tmp_path):
    df = pd.DataFrame({"id": [1], "mes": [202601], "target": [0], "nota": ["x"]})
    cfg = ConfigPipeline(columna_id="id", columna_tiempo="mes", columnas_conservadas=["nota"],
                         exportar_dataset_final=False, ruta_dataset_final=str(tmp_path / "final.csv"))
    assert _exportar_dataset_final(df, cfg, []) is None
    cfg.exportar_dataset_final = True
    cfg.columnas_conservadas = ["ausente"]
    assert _exportar_dataset_final(df, cfg, []) is None
    assert not (tmp_path / "final.csv").exists()


def test_excluidas_no_se_exportan(tmp_path):
    df = pd.DataFrame({"id": [1], "mes": [202601], "target": [0],
                       "senal": [1.5], "nota": ["guardar"], "secreto": ["omitir"]})
    cfg = ConfigPipeline(columna_id="id", columna_tiempo="mes",
                         columnas_conservadas=["nota"], columnas_excluidas=["secreto"],
                         ruta_dataset_final=str(tmp_path / "final.csv"))
    cfg.validar()
    ruta = _exportar_dataset_final(df, cfg, ["senal", "secreto"])
    assert pd.read_csv(ruta).columns.tolist() == ["id", "mes", "target", "senal", "nota"]
    assert cfg.rol_de("secreto") == "EXCLUIDA_TOTAL"
    assert "secreto" in cfg.columnas_no_candidatas


def test_listas_manuales_no_admiten_conflictos():
    from featsel.config import ErrorConfiguracion

    with pytest.raises(ErrorConfiguracion, match="conservadas y excluidas"):
        ConfigPipeline(columnas_conservadas=["nota"], columnas_excluidas=["nota"]).validar()
    with pytest.raises(ErrorConfiguracion, match="Columnas de rol"):
        ConfigPipeline(columnas_excluidas=[ConfigPipeline().columna_id]).validar()


def test_excluidas_se_retiran_antes_del_analisis(monkeypatch):
    from featsel import pipeline

    df = pd.DataFrame({"id": [1, 2], "mes": [202601, 202602], "target": [0, 1],
                       "nota": ["a", "b"], "secreto": [12, 34], "senal": [1.0, 2.0]})
    cfg = ConfigPipeline(columna_id="id", columna_tiempo="mes",
                         columnas_conservadas=["nota"], columnas_excluidas=["secreto"],
                         usar_feature_engineering=False)
    monkeypatch.setattr(pipeline.io_utils, "cargar_dataset", lambda cfg: df)

    class DiagnosticoAlcanzado(Exception):
        pass

    def verificar(entrada, *args):
        assert "secreto" not in entrada.columns
        assert "nota" in entrada.columns
        assert "senal" in entrada.columns
        raise DiagnosticoAlcanzado

    monkeypatch.setattr(pipeline.fase0_diagnostico, "ejecutar", verificar)
    with pytest.raises(DiagnosticoAlcanzado):
        pipeline.ejecutar(cfg)
