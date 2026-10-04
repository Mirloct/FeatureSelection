"""Pruebas adversarias del contrato KDE.

Data sources / inputs: paneles sinteticos en memoria y config.yaml.
Created: 2026-10-04
Last modified: 2026-10-04
Changelog:
- 2026-10-04: cobertura de columnas, colisiones, infinitos y causalidad.
"""
import numpy as np
import pandas as pd
import pytest

from featsel.config import ConfigPipeline, ErrorConfiguracion
from featsel.feature_engineering_anomalias import ejecutar, ErrorFeatureEngineering


def config(**kwargs):
    values = dict(usar_feature_engineering=True, columna_id="id", columna_tiempo="mes",
                  context_vars=["grupo"], behavior_vars=["valor"], min_group_size=2,
                  min_personal_history=2, temporal_windows=[2], bandwidth_method=0.5,
                  preparar_modelos_anomalia=False)
    values.update(kwargs)
    cfg = ConfigPipeline(**values)
    cfg.validar()
    return cfg


def panel():
    return pd.DataFrame(dict(id=[1, 2]*4, mes=np.repeat([1, 2, 3, 4], 2),
                             grupo=["A"]*8, valor=[1., 2., 3., 4., 5., 6., 7., 8.]))


@pytest.mark.parametrize("column", ["__orden_original", "__ctx__grupo"])
def test_no_pierde_columnas_auxiliares_del_usuario(column):
    df = panel()
    df[column] = np.arange(len(df)) + 100
    result = ejecutar(df, config()).dataframe
    pd.testing.assert_frame_equal(result[df.columns], df)


def test_rechaza_nombres_de_features_ambiguos():
    df = panel().assign(**{"VALOR": np.arange(8.)})
    with pytest.raises(ErrorFeatureEngineering, match="colision"):
        ejecutar(df, config(behavior_vars=["valor", "VALOR"]))


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_rechaza_infinitos_con_error_de_dominio(value):
    df = panel()
    df.loc[0, "valor"] = value
    with pytest.raises(ErrorFeatureEngineering, match="finitos"):
        ejecutar(df, config())


def test_nulos_no_se_mezclan_con_categoria_literal_missing():
    df = panel()
    df["grupo"] = [None, "__MISSING__"]*4
    result = ejecutar(df, config()).dataframe
    cols = result.filter(like="fe__kde").columns
    assert result.loc[result.mes.eq(2), cols].isna().all().all()
    assert result.loc[result.mes.eq(3), cols].notna().all().all()


def test_columnas_elegidas_y_futuro_aislados():
    df = panel().assign(ignorada=np.arange(8.))
    cfg = config()
    base = ejecutar(df, cfg)
    changed = df.copy()
    changed["ignorada"] = 1e9
    changed.loc[changed.mes.eq(4), "valor"] = 1e6
    result = ejecutar(changed, cfg)
    cols = base.catalogo.columna.tolist()
    pd.testing.assert_frame_equal(base.dataframe.loc[df.mes.lt(4), cols],
                                  result.dataframe.loc[df.mes.lt(4), cols])
    assert set(base.catalogo.behavior_var) == {"valor"}
    assert set(base.catalogo.loc[base.catalogo.familia.eq("KDE"), "referencia"]) == {"grupo"}


def test_indice_repetido_y_filas_desordenadas():
    df = panel().sample(frac=1, random_state=2)
    df.index = [0]*len(df)
    result = ejecutar(df, config()).dataframe
    expected = ejecutar(panel(), config()).dataframe
    pd.testing.assert_frame_equal(result.sort_values(["mes", "id"]).reset_index(drop=True), expected)


@pytest.mark.parametrize("kernel", ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"])
@pytest.mark.parametrize("mode,ncols", [("joint", 2), ("marginal", 2), ("joint_and_marginal", 4)])
def test_kernels_y_modos_activos(kernel, mode, ncols):
    result = ejecutar(panel(), config(kernel_type=kernel, reference_mode=mode))
    cols = result.catalogo.loc[result.catalogo.familia.eq("KDE"), "columna"].tolist()
    assert len(cols) == ncols
    assert np.isfinite(result.dataframe.loc[result.dataframe.mes.gt(1), cols]).all().all()


@pytest.mark.parametrize("param,value", [("epsilon", np.nan), ("epsilon", np.inf),
    ("bandwidth_method", np.nan), ("bandwidth_method", "inf"),
    ("bandwidth_grid", [np.inf]), ("temporal_windows", [2.5]),
    ("temporal_windows", ["bad"])])
def test_parametros_invalidos_fallan_antes_del_kernel(param, value):
    with pytest.raises(ErrorConfiguracion, match=param):
        config(**{param: value})
