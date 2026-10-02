"""
Pruebas de causalidad y contrato del feature engineering de anomalias.

Data sources / inputs: panel sintetico construido en memoria por cada prueba.
Created: 2026-10-01
Last modified: 2026-10-01
Changelog:
- 2026-10-01: pruebas iniciales de interruptor, no-look-ahead y matriz IF/VAE.
- 2026-10-01: sys.path propio para correr con pytest, unittest discover o
  ejecucion directa sin depender solo de tests/conftest.py (ver run_pipeline.py).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from featsel.config import ConfigPipeline
from featsel.feature_engineering_anomalias import ejecutar, preparar_para_modelos


class FeatureEngineeringAnomaliasTest(unittest.TestCase):
    """Cubre invariantes que, si fallan, producirian fuga temporal."""

    @staticmethod
    def _panel() -> pd.DataFrame:
        rng = np.random.default_rng(42)
        filas = []
        for mes in pd.date_range("2024-01-01", periods=10, freq="MS"):
            for entidad in range(12):
                filas.append({
                    "customer_id": f"C{entidad:02d}",
                    "month": mes,
                    "age_group": "A" if entidad < 6 else "B",
                    "job_position": "X" if entidad % 2 else "Y",
                    "credit_count": float(rng.poisson(3)),
                    "fraud_flag": int(entidad == 0 and mes.month > 7),
                })
        return pd.DataFrame(filas)

    @staticmethod
    def _config(**overrides) -> ConfigPipeline:
        base = dict(
            columna_target="fraud_flag",
            columna_id="customer_id",
            columna_tiempo="month",
            usar_feature_engineering=True,
            context_vars=["age_group", "job_position"],
            behavior_vars=["credit_count"],
            min_group_size=3,
            min_personal_history=6,
            temporal_windows=[3],
            bandwidth_method="time_safe_cv",
            bandwidth_grid=[0.2, 0.5],
            meses_holdout_anomalia=2,
        )
        base.update(overrides)
        cfg = ConfigPipeline(**base)
        cfg.validar()
        return cfg

    def test_interruptor_apagado_no_modifica_dataframe(self) -> None:
        df = self._panel()
        cfg = self._config(usar_feature_engineering=False)
        resultado = ejecutar(df, cfg)
        pd.testing.assert_frame_equal(resultado.dataframe, df)
        self.assertTrue(resultado.catalogo.empty)

    def test_primer_mes_y_futuro_no_contaminan_features(self) -> None:
        df = self._panel()
        cfg = self._config()
        original = ejecutar(df, cfg)
        columnas_fe = original.catalogo["columna"].tolist()
        primer_mes = original.dataframe["month"].eq(original.dataframe["month"].min())
        self.assertTrue(original.dataframe.loc[primer_mes, columnas_fe].isna().all().all())

        alterado = df.copy()
        ultimo = alterado["month"].max()
        alterado.loc[alterado["month"].eq(ultimo), "credit_count"] = 1_000_000.0
        recalculado = ejecutar(alterado, cfg)
        antes_del_ultimo = original.dataframe["month"].lt(ultimo)
        pd.testing.assert_frame_equal(
            original.dataframe.loc[antes_del_ultimo, columnas_fe].reset_index(drop=True),
            recalculado.dataframe.loc[antes_del_ultimo, columnas_fe].reset_index(drop=True),
        )

    def test_matriz_if_vae_no_tiene_nulos_y_separa_holdout(self) -> None:
        df = self._panel()
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "matriz.csv"
            cfg = self._config(ruta_matriz_anomalias=str(ruta))
            resultado = ejecutar(df, cfg)
            seleccionadas = resultado.catalogo["columna"].tolist()[:4] + ["age_group"]
            prep = preparar_para_modelos(resultado.dataframe, cfg, seleccionadas)
            self.assertEqual(int(prep.matriz[seleccionadas].isna().sum().sum()), 0)
            self.assertEqual(set(prep.matriz["split_temporal"]), {"TRAIN", "HOLDOUT"})
            self.assertTrue(ruta.is_file())


if __name__ == "__main__":
    unittest.main()
