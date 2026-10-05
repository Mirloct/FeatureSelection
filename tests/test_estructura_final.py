"""Tercera iteracion: contratos reales de pipeline, Excel y CSV.

Data sources / inputs: CSV sinteticos temporales y config.yaml.
Created: 2026-10-04
Last modified: 2026-10-04
Changelog:
- 2026-10-04: se retiraron los casos de feature engineering e IF/VAE
  (implementados ahora en el codigo base de entrada del usuario).
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


@pytest.mark.parametrize("supervisado", [False, True])
def test_pipeline_estructura_y_trazabilidad(tmp_path, supervisado):
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
                         usar_boruta=False, laplacian_n_permutaciones=19,
                         excluir_sospecha_fuga=False)
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


def test_esquema_final_cubre_todo_el_yaml_central():
    assert {f.name for f in fields(ConfigPipeline)} == set(_leer_yaml(CONFIG_PREDETERMINADA))
    cfg = ConfigPipeline()
    cfg.validar()
