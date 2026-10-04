"""Tercera iteracion: contratos reales de pipeline, Excel, CSV e IF/VAE.

Data sources / inputs: CSV sinteticos temporales y config.yaml.
Created: 2026-10-04
Last modified: 2026-10-04
Changelog:
- 2026-10-04: verifica ramas con/sin target y FE encendido/apagado.
"""
import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook
from dataclasses import fields

from featsel.config import ConfigPipeline
from featsel.config import CONFIG_PREDETERMINADA, _leer_yaml
from featsel.pipeline import ejecutar
from featsel.feature_engineering_anomalias import preparar_para_modelos


@pytest.mark.parametrize("supervisado", [False, True])
@pytest.mark.parametrize("fe_activo", [False, True])
def test_pipeline_estructura_y_trazabilidad(tmp_path, supervisado, fe_activo):
    rng = np.random.default_rng(19)
    rows = []
    for mes in range(1, 7):
        for entidad in range(12):
            saldo = float(entidad + rng.normal(0, 2))
            rows.append(dict(id=entidad, mes=mes, target=int(saldo > 5), saldo=saldo,
                             clon=saldo, zona="A" if entidad < 6 else "B",
                             secreto="retirar", **{"=auditoria": f"{entidad:04d}"}))
    df = pd.DataFrame(rows)
    if not supervisado:
        df = df.drop(columns="target")
    csv = tmp_path / "input.csv"
    df.to_csv(csv, index=False)
    cfg = ConfigPipeline(ruta_dataset=str(csv), columna_id="id", columna_tiempo="mes",
                         ruta_salida_excel=str(tmp_path / "bitacora.xlsx"),
                         columnas_conservadas=["=auditoria"], columnas_excluidas=["secreto"],
                         usar_boruta=False, usar_feature_engineering=fe_activo,
                         usar_estandarizacion_puesto=False, context_vars=["zona"],
                         behavior_vars=["saldo"], min_group_size=3, min_personal_history=2,
                         temporal_windows=[2], bandwidth_method=.5,
                         laplacian_n_permutaciones=19, excluir_sospecha_fuga=False,
                         preparar_modelos_anomalia=False)
    cfg.validar()
    result = ejecutar(cfg)
    final = pd.read_csv(result["ruta_dataset_final"])
    selected = result["variables_seleccionadas"]
    expected = ["id", "mes"] + (["target"] if supervisado else []) + selected + ["=auditoria"]
    assert final.columns.tolist() == expected
    assert len(final) == len(df)
    assert final[["id", "mes"]].values.tolist() == df[["id", "mes"]].values.tolist()
    assert not {"id", "mes", "target", "=auditoria", "secreto"}.intersection(selected)
    assert not {"saldo", "clon"}.issubset(selected)
    assert result["modo_supervisado"] is supervisado
    catalog = result["feature_engineering_catalogo"]
    assert catalog.empty is (not fe_activo)
    if fe_activo:
        assert set(catalog.behavior_var) == {"saldo"}
        assert catalog.columna.is_unique
    wb = load_workbook(result["ruta_excel"])
    assert "06_Seleccion_Final" in wb.sheetnames
    assert ("03_Bivariado" in wb.sheetnames) is supervisado
    assert ("03_Relevancia_NoSuperv" in wb.sheetnames) is (not supervisado)
    assert "05_Boruta" not in wb.sheetnames
    # Los encabezados del usuario que empiezan por '=' siguen siendo texto.
    matches = [cell for ws in wb for row in ws for cell in row if cell.value == "=auditoria"]
    assert matches
    assert all(cell.data_type == "s" for cell in matches)
    wb.close()


def test_if_vae_holdout_no_cambia_train_ni_parametros(tmp_path):
    df = pd.DataFrame(dict(id=np.tile([1, 2], 6), mes=np.repeat(np.arange(1, 7), 2),
                           saldo=np.arange(12.), categoria=["A", "B"]*6))
    cfg = ConfigPipeline(columna_id="id", columna_tiempo="mes", usar_feature_engineering=True,
                         meses_holdout_anomalia=2, ruta_matriz_anomalias=str(tmp_path / "matrix.csv"))
    base = preparar_para_modelos(df, cfg, ["saldo", "categoria"])
    changed = df.copy()
    changed.loc[changed.mes.gt(4), "saldo"] = 1e6
    changed.loc[changed.mes.gt(4), "categoria"] = "NUEVA"
    other = preparar_para_modelos(changed, cfg, ["saldo", "categoria"])
    mask = base.matriz.split_temporal.eq("TRAIN")
    pd.testing.assert_frame_equal(base.matriz.loc[mask], other.matriz.loc[mask])
    pd.testing.assert_frame_equal(base.esquema, other.esquema)
    assert np.isfinite(other.matriz[["saldo", "categoria"]]).all().all()
    assert base.matriz.columns.tolist() == ["id", "mes", "split_temporal", "saldo", "categoria"]
    assert base.matriz.split_temporal.value_counts().to_dict() == {"TRAIN": 8, "HOLDOUT": 4}


def test_esquema_final_cubre_todo_el_yaml_central():
    assert {f.name for f in fields(ConfigPipeline)} == set(_leer_yaml(CONFIG_PREDETERMINADA))
    cfg = ConfigPipeline()
    cfg.validar()
    assert cfg.usar_feature_engineering is False
