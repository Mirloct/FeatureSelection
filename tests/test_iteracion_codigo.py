"""Segunda iteracion: decisiones adversarias y estadistica de permutacion.

Data sources / inputs: paneles sinteticos en memoria y config.yaml.
Created: 2026-10-04
Last modified: 2026-10-04
Changelog:
- 2026-10-04: pruebas de p-valores, grafos y decisiones multivariadas.
"""
import numpy as np
import pandas as pd
import pytest
from featsel import metricas as m, fase3_multivariado as fase3
from featsel.config import ConfigPipeline


def test_permutacion_no_produce_p_cero_y_respeta_resolucion():
    x = np.arange(35, dtype=float)
    df = pd.DataFrame(dict(a=x, b=x))
    table = m.laplacian_score_con_piso_ruido(df, n_permutaciones=9, semilla=7)
    assert (table.p_valor_estructura >= 1/10).all()
    assert table.flg_supera_ruido.eq(0).all()


def test_permutacion_formula_montecarlo(monkeypatch):
    values = iter([.2, .1, .3, .4, .5, .6])
    monkeypatch.setattr(m, "_puntaje_laplaciano_vector", lambda *args: next(values))
    df = pd.DataFrame(dict(a=np.arange(20.), b=np.arange(20.)))
    table = m.laplacian_score_con_piso_ruido(df, n_permutaciones=2)
    np.testing.assert_allclose(table.p_valor_estructura, [2/3, 1/3])


def test_grafo_no_autoconecta_observaciones_duplicadas():
    x = np.array([[0.], [0.], [0.], [1.], [2.]])
    S = m._grafo_vecinos(x, 2)
    assert np.all(S.diagonal() == 0)
    np.testing.assert_allclose(S.toarray(), S.T.toarray())


@pytest.mark.parametrize("excluir", [False, True])
def test_vif_infinito_se_reporta_y_se_puede_excluir(excluir):
    rng = np.random.default_rng(9)
    a, b = rng.normal(size=(2, 150))
    df = pd.DataFrame(dict(a=a, b=b, suma=a+b))
    cfg = ConfigPipeline(umbral_correlacion=.999, excluir_por_vif=excluir)
    tipos = {c: "NUMERICA" for c in df}
    scores = pd.DataFrame(dict(columna=list(df), score_compuesto=[.3,.5,.8]))
    multi, _, _ = fase3.ejecutar(df, cfg, tipos, list(df), scores)
    if excluir:
        assert multi.flg_exclusion_multivariada.sum() == 1
        assert len(fase3.obtener_seleccion_final(multi)) == 2
    else:
        assert multi.flg_vif_alto.eq(1).all()
        assert np.isinf(multi.vif).all()


def test_empate_score_e_iv_se_resuelve_por_gini():
    df = pd.DataFrame(dict(a=np.arange(30.), b=np.arange(30.)))
    scores = pd.DataFrame(dict(columna=["a", "b"], score_compuesto=[.5,.5],
                               iv=[.1,.1], gini=[.2,.8]))
    multi, _, _ = fase3.ejecutar(df, ConfigPipeline(), {c:"NUMERICA" for c in df}, list(df), scores)
    assert fase3.obtener_seleccion_final(multi) == ["b"]


def test_vif_singular_no_contamina_variable_independiente():
    # Vectores ortogonales: z no comparte la dependencia entre a y b.
    a = np.tile([-1., -1., 1., 1.], 20)
    z = np.tile([-1., 1., -1., 1.], 20)
    result = m.calcular_vif(pd.DataFrame(dict(a=a, b=2*a, z=z)))
    assert np.isinf(result[["a", "b"]]).all()
    assert result.z == pytest.approx(1)


def test_alpha_configurado_no_se_recorta(monkeypatch):
    values = iter([.2, .1, .3, .4, .5, .6])
    monkeypatch.setattr(m, "_puntaje_laplaciano_vector", lambda *args: next(values))
    df = pd.DataFrame(dict(a=np.arange(20.), b=np.arange(20.)))
    table = m.laplacian_score_con_piso_ruido(df, n_permutaciones=2, alpha=.9)
    assert table.flg_supera_ruido.eq(1).all()


def test_score_mayor_no_se_trata_como_empate_aproximado():
    df = pd.DataFrame(dict(a=np.arange(30.), b=np.arange(30.)))
    scores = pd.DataFrame(dict(columna=["a", "b"], score_compuesto=[.5, .5000001],
                               iv=[.9,.1], gini=[.5,.5]))
    multi, _, _ = fase3.ejecutar(df, ConfigPipeline(), {c:"NUMERICA" for c in df}, list(df), scores)
    assert fase3.obtener_seleccion_final(multi) == ["b"]
