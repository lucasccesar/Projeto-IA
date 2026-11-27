import numpy as np
import pandas as pd

from src.data.dataset_final import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.knn import DISTANCIAS, VALORES_K
from src.models.knn_pso import buscar_hiperparametros_pso


def _conjunto(n, semente):
    rng = np.random.default_rng(semente)
    df = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLUMNS))),
        columns=list(FEATURE_COLUMNS),
    )
    df[TARGET_COLUMN] = 20 + 5 * df["mediaKills_hist"] + rng.normal(scale=0.5, size=n)
    return df


def test_pso_retorna_configuracoes_validas_e_historico_de_convergencia():
    busca, historico = buscar_hiperparametros_pso(
        _conjunto(250, 0),
        _conjunto(80, 1),
        tamanho_enxame=4,
        num_iteracoes=5,
        semente=42,
    )

    assert not busca.empty
    assert set(busca["k"]).issubset(set(VALORES_K))
    assert set(busca["distancia"]).issubset(set(DISTANCIAS))
    assert busca["MAE_validacao"].is_monotonic_increasing
    assert list(historico["iteracao"]) == list(range(6))
    assert historico["melhor_MAE_validacao"].is_monotonic_decreasing
    assert historico["configuracoes_avaliadas"].is_monotonic_increasing
    assert len(busca) <= len(VALORES_K) * len(DISTANCIAS)


def test_pso_e_reprodutivel_com_a_mesma_semente():
    treino, validacao = _conjunto(250, 0), _conjunto(80, 1)

    busca_a, historico_a = buscar_hiperparametros_pso(
        treino, validacao, tamanho_enxame=4, num_iteracoes=5, semente=42
    )
    busca_b, historico_b = buscar_hiperparametros_pso(
        treino, validacao, tamanho_enxame=4, num_iteracoes=5, semente=42
    )

    pd.testing.assert_frame_equal(busca_a, busca_b)
    pd.testing.assert_frame_equal(historico_a, historico_b)
