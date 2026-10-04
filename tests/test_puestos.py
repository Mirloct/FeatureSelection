"""Regresion de normalizacion de puestos, fallbacks y uso real en KDE.

Data sources / inputs: paneles sinteticos y CSV temporal con encoding cp1252.
Created: 2026-10-02
Last modified: 2026-10-02
Changelog:
- 2026-10-02: prioridad, aliases, texto danado, KDE causal y exportacion.
"""

import numpy as np
import pandas as pd
import pytest

from featsel.config import ConfigPipeline
from featsel import puestos, feature_engineering_anomalias as fe, io_utils, pipeline


@pytest.mark.parametrize("valor, esperado", [
    ("GERENTE", "GERENTE"), ("GTE", "GERENTE"), ("Gte", "GERENTE"),
    ("GERENTE DE negocios", "GERENTE"), ("gerente adjunto", "GERENTE ADJUNTO"),
    ("GTEADJ", "GERENTE ADJUNTO"), ("GTE.ADJ", "GERENTE ADJUNTO"),
    ("asistente de operaciones", "ASISTENTE"), ("asistrnte", "ASISTENTE"),
    ("analista senior riesgos", "ANALISTA SENIOR"), ("ANL JR", "ANALISTA JUNIOR"),
    ("Técnico de soporte", "TECNICO"), ("TÃ©cnico de soporte", "TECNICO"),
    ("TÃƒÂ©cnico", "TECNICO"), ("T�cnico", "TECNICO"),
    ("ANAL�STA", "ANALISTA"), ("sub gte comercial", "SUBGERENTE"),
    ("  , . -  vendedor tienda", "VENDEDOR"), ("otro puesto largo", "OTRO"),
    ("-", "SIN INFORMACION"), (",", "SIN INFORMACION"), (".", "SIN INFORMACION"),
    ("123", "SIN INFORMACION"), (None, "SIN INFORMACION"),
    (np.nan, "SIN INFORMACION"), (pd.NA, "SIN INFORMACION"),
    ("N/A", "SIN INFORMACION"), ("sin información", "SIN INFORMACION"),
])
def test_normalizacion(valor, esperado):
    assert puestos.agrupar_puesto(valor, "SIN INFORMACION") == esperado


@pytest.mark.parametrize("fuentes", [
    {"desposicioncolaborador": ["GTE", "."]},
    {"despuestocolaborador": ["GTE", "."]},
    {"despuestocolaborador": ["GTE", "."], "desposicioncolaborador": ["analista", "asistente"]},
    {" DESPUESTOCOLABORADOR ": ["GTE", "."]},
])
def test_prioridad_contexto_y_exportacion(tmp_path, fuentes):
    df = pd.DataFrame({"id": [1, 2], "mes": [202601, 202601], "target": [0, 1], **fuentes})
    cfg = ConfigPipeline(columna_id="id", columna_tiempo="mes", context_vars=["job_position"],
                         ruta_dataset_final=str(tmp_path / "final.csv"))
    resultado, efectivo = puestos.preparar(df, cfg)
    agrupado = "despuestocolaboradoragrupado"
    assert resultado[agrupado].tolist() == ["GERENTE", "SIN INFORMACION"]
    assert efectivo.context_vars == [agrupado]
    assert agrupado in efectivo.columnas_conservadas
    assert cfg.context_vars == ["job_position"]
    assert agrupado not in cfg.columnas_conservadas
    if len(fuentes) == 2:
        assert "desposicioncolaborador" not in resultado.columns
    ruta = pipeline._exportar_dataset_final(resultado, efectivo, [])
    exportado = pd.read_csv(ruta)
    assert exportado.columns[-1] == agrupado
    assert exportado[agrupado].tolist() == resultado[agrupado].tolist()


def test_sin_fuente_o_desactivado_no_modifica():
    df = pd.DataFrame({"otra": [1]})
    cfg = ConfigPipeline()
    resultado, efectivo = puestos.preparar(df, cfg)
    assert resultado is df and efectivo is cfg
    df["despuestocolaborador"] = "GTE"
    cfg.usar_estandarizacion_puesto = False
    resultado, efectivo = puestos.preparar(df, cfg)
    assert resultado is df and efectivo is cfg


def test_encoding_csv_real(tmp_path):
    ruta = tmp_path / "cp1252.csv"
    ruta.write_bytes("despuestocolaborador\nTécnico\n".encode("cp1252"))
    cfg = ConfigPipeline(ruta_dataset=str(ruta))
    df = io_utils.cargar_dataset(cfg)
    resultado, _ = puestos.preparar(df, cfg)
    assert resultado["despuestocolaboradoragrupado"].tolist() == ["TECNICO"]


def test_kde_usa_grupo_y_conserva_causalidad():
    filas = []
    for mes in pd.date_range("2024-01-01", periods=5, freq="MS"):
        for entidad in range(6):
            filas.append({"id": entidad, "mes": mes, "zona": "LIMA",
                          "despuestocolaborador": "Gte" if entidad % 2 else "GERENTE DE",
                          "saldo": float(entidad + mes.month)})
    df = pd.DataFrame(filas)
    cfg = ConfigPipeline(columna_id="id", columna_tiempo="mes", usar_feature_engineering=True,
                         context_vars=["zona", "despuestocolaborador"], behavior_vars=["saldo"],
                         min_group_size=3, min_personal_history=2, temporal_windows=[2],
                         bandwidth_method=0.5)
    preparado, efectivo = puestos.preparar(df, cfg)
    assert efectivo.context_vars == ["zona", "despuestocolaboradoragrupado"]
    resultado = fe.ejecutar(preparado, efectivo)
    cols = [c for c in resultado.dataframe if "marginal_despuestocolaboradoragrupado" in c]
    assert cols
    assert resultado.dataframe[cols].notna().any().any()
    cambiado = df.copy()
    ultimo = df["mes"].max()
    cambiado.loc[cambiado["mes"].eq(ultimo), "despuestocolaborador"] = "analista"
    cambiado.loc[cambiado["mes"].eq(ultimo), "saldo"] = 9999
    preparado2, cfg2 = puestos.preparar(cambiado, cfg)
    resultado2 = fe.ejecutar(preparado2, cfg2)
    anteriores = df["mes"].lt(ultimo)
    pd.testing.assert_frame_equal(resultado.dataframe.loc[anteriores, cols],
                                  resultado2.dataframe.loc[anteriores, cols])


@pytest.mark.parametrize("con_target", [True, False])
def test_pipeline_completo_exporta_agrupado(tmp_path, con_target):
    rng = np.random.default_rng(42)
    filas = []
    for mes in pd.date_range("2024-01-01", periods=5, freq="MS"):
        for entidad in range(8):
            filas.append({"id": entidad, "mes": mes,
                          "target": entidad % 2, "despuestocolaborador": "GTE",
                          "desposicioncolaborador": "analista", "saldo": float(rng.normal(20, 5))})
    df = pd.DataFrame(filas)
    if not con_target:
        df = df.drop(columns="target")
    entrada = tmp_path / "panel.csv"
    df.to_csv(entrada, index=False)
    cfg = ConfigPipeline(ruta_dataset=str(entrada), columna_id="id", columna_tiempo="mes",
                         ruta_salida_excel=str(tmp_path / "bitacora.xlsx"),
                         usar_boruta=False, usar_feature_engineering=True,
                         context_vars=["desposicioncolaborador"], behavior_vars=["saldo"],
                         min_group_size=3, min_personal_history=2, temporal_windows=[2],
                         bandwidth_method=0.5, laplacian_n_permutaciones=5,
                         preparar_modelos_anomalia=False)
    cfg.validar()
    resultados = pipeline.ejecutar(cfg)
    final = pd.read_csv(resultados["ruta_dataset_final"])
    assert len(final) == len(df)
    assert final.columns[-1] == "despuestocolaboradoragrupado"
    assert final["despuestocolaboradoragrupado"].eq("GERENTE").all()
    assert "desposicioncolaborador" not in final.columns
    assert resultados["cfg"].context_vars == ["despuestocolaboradoragrupado"]
    assert (resultados["feature_engineering_catalogo"]["familia"] == "KDE").any()
    assert "despuestocolaboradoragrupado" not in resultados["variables_seleccionadas"]
