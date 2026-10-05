"""Referencias independientes para los fundamentos estadisticos.

Data sources / inputs: muestras sinteticas en memoria y config.yaml.
Created: 2026-10-04
Last modified: 2026-10-04
Changelog:
- 2026-10-04: se retiro la prueba de KDE (feature engineering implementado
  ahora en el codigo base de entrada del usuario).
- 2026-10-04: primera iteracion; formulas de IV, Gini, PSI, eta y VIF.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from featsel import metricas as m


def test_woe_iv_conteos_y_normalizacion():
    bins = pd.Series(["A"]*10 + ["B"]*10)
    y = np.array([1]*2+[0]*8+[1]*8+[0]*2)
    table, iv = m.tabla_woe(bins, y, correccion=0.5)
    p = np.array([2.5, 8.5])/11
    q = p[::-1]
    np.testing.assert_allclose(table.woe, np.log(q/p))
    assert iv == pytest.approx(np.sum((q-p)*np.log(q/p)))
    assert table.dist_evento.sum() == pytest.approx(1)
    assert table.dist_no_evento.sum() == pytest.approx(1)


def test_iv_independencia_y_simetria_evento():
    bins = pd.Series(["A"]*10+["B"]*10)
    y = np.tile([0, 1], 10)
    assert m.tabla_woe(bins, y)[1] == pytest.approx(0)
    y = np.array([0]*8+[1]*2+[0]*2+[1]*8)
    assert m.tabla_woe(bins, y)[1] == pytest.approx(m.tabla_woe(bins, 1-y)[1])


def test_gini_auc_comparacion_de_pares_con_empates():
    score = pd.Series([0, 1, 1, 2, 3, 0, 2, 3, 3, 4])
    y = pd.Series([0]*5+[1]*5)
    positives, negatives = score[y.eq(1)], score[y.eq(0)]
    auc = np.mean([float(p>n)+0.5*float(p==n) for p in positives for n in negatives])
    absolute, signed = m.calcular_gini(score, y)
    assert signed == pytest.approx(2*auc-1)
    assert absolute == pytest.approx(abs(signed))
    assert m.calcular_gini(-score, y)[1] == pytest.approx(-signed)


def test_psi_divergencia_identidad_y_simetria():
    a, b = pd.Series(["A"]*8+["B"]*2), pd.Series(["A"]*2+["B"]*8)
    eps = 1e-6
    p = (np.array([.8, .2])+eps)/(1+2*eps)
    q = p[::-1]
    assert m.calcular_psi(a, b) == pytest.approx(np.sum((p-q)*np.log(p/q)))
    assert m.calcular_psi(a, a) == pytest.approx(0)
    assert m.calcular_psi(a, b) == pytest.approx(m.calcular_psi(b, a))


def test_eta_y_cramer_separacion_perfecta():
    cat = pd.Series(["A"]*10+["B"]*10)
    numeric = pd.Series([0.]*10+[10.]*10)
    assert m.razon_correlacion(numeric, cat) == pytest.approx(1)
    assert m.v_de_cramer(cat, cat, correccion_sesgo=False) == pytest.approx(1)


def test_vif_contra_regresiones_independientes():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(150, 3))
    x[:, 1] += .7*x[:, 0]
    expected = []
    for j in range(3):
        predictors = np.column_stack([np.ones(len(x)), np.delete(x, j, axis=1)])
        fitted = predictors @ np.linalg.lstsq(predictors, x[:, j], rcond=None)[0]
        expected.append(np.sum((x[:, j]-x[:, j].mean())**2)/np.sum((x[:, j]-fitted)**2))
    np.testing.assert_allclose(m.calcular_vif(pd.DataFrame(x)), expected, rtol=1e-10)


def test_vif_colinealidad_exacta_es_infinito():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=(2, 100))
    result = m.calcular_vif(pd.DataFrame(dict(a=a, b=b, suma=a+b)))
    assert np.isinf(result).all()


def test_laplaciano_energia_por_aristas_e_invariancia():
    S = np.array([[0, 1, 0], [1, 0, 2], [0, 2, 0]], dtype=float)
    d = S.sum(axis=1)
    f = np.array([1., 3., 2.])
    centered = f-np.average(f, weights=d)
    energy = sum(S[i, j]*(f[i]-f[j])**2 for i in range(3) for j in range(i+1, 3))
    expected = energy/np.sum(d*centered**2)
    L = sparse.csr_matrix(np.diag(d)-S)
    assert m._puntaje_laplaciano_vector(f, d, L) == pytest.approx(expected)
    assert m._puntaje_laplaciano_vector(3*f+10, d, L) == pytest.approx(expected)
