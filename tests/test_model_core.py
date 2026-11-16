import unittest

import pandas as pd

from model import (
    aplicar_dixon_coles,
    calcular_1x2,
    calcular_mercados_adicionales,
    construir_combinada_pro,
    construir_top_picks_pro,
    matriz_probabilidades,
    recalibrar_matriz_1x2,
)


class ModelCoreTests(unittest.TestCase):
    def test_recalibrated_matrix_matches_1x2_targets(self):
        matriz = matriz_probabilidades(1.4, 0.9)
        matriz = aplicar_dixon_coles(matriz, 1.4, 0.9)
        ajustada = recalibrar_matriz_1x2(matriz, 0.56, 0.25, 0.19)
        local, empate, visitante = calcular_1x2(ajustada)
        self.assertAlmostEqual(ajustada.sum(), 1.0, places=10)
        self.assertAlmostEqual(local, 0.56, places=8)
        self.assertAlmostEqual(empate, 0.25, places=8)
        self.assertAlmostEqual(visitante, 0.19, places=8)

    def test_related_markets_come_from_one_coherent_matrix(self):
        matriz = recalibrar_matriz_1x2(matriz_probabilidades(1.8, 1.1), 0.58, 0.23, 0.19)
        mercados = calcular_mercados_adicionales(matriz)
        local, empate, visitante = calcular_1x2(matriz)
        self.assertAlmostEqual(mercados["local_empate"], local + empate, places=10)
        self.assertAlmostEqual(mercados["visitante_empate"], visitante + empate, places=10)
        self.assertGreaterEqual(mercados["over_15"], mercados["over_25"])
        self.assertGreaterEqual(mercados["under_45"], mercados["under_35"])
        self.assertGreaterEqual(mercados["local_anota"], mercados["local_anota_2"])
        self.assertGreaterEqual(mercados["visitante_anota"], mercados["visitante_anota_2"])

    def test_top_picks_are_real_odds_and_independent_fixtures(self):
        df = pd.DataFrame([
            {"fixture_id": "a", "partido": "A vs B", "decision_simple": "Apostar", "odds_source_pick": "real", "probabilidad_pick": 62, "cuota_pick": 1.9, "edge_pick": 4, "ev_pick": 8, "score_final": 55, "confianza": 70, "riesgo_simple": "Riesgo medio", "pick_recomendado": "Gana el local", "mercado_recomendado": "1X2"},
            {"fixture_id": "a", "partido": "A vs B", "decision_simple": "Apostar", "odds_source_pick": "real", "probabilidad_pick": 70, "cuota_pick": 1.4, "edge_pick": 2, "ev_pick": 3, "score_final": 40, "confianza": 71, "riesgo_simple": "Riesgo medio", "pick_recomendado": "Local anota", "mercado_recomendado": "Equipo marca"},
            {"fixture_id": "c", "partido": "C vs D", "decision_simple": "Apostar", "odds_source_pick": "real", "probabilidad_pick": 60, "cuota_pick": 2.0, "edge_pick": 3, "ev_pick": 6, "score_final": 50, "confianza": 65, "riesgo_simple": "Riesgo medio", "pick_recomendado": "Gana el visitante", "mercado_recomendado": "1X2"},
        ])
        picks = construir_top_picks_pro(df)
        self.assertEqual(len(picks), 2)
        self.assertEqual(picks["fixture_id"].nunique(), 2)
        combinada = construir_combinada_pro(picks)
        self.assertIsNotNone(combinada)
        self.assertEqual(len(combinada["selecciones"]), 2)


if __name__ == "__main__":
    unittest.main()
