import numpy as np
import pandas as pd

from src.data.dataset_final import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.arvore_decisao import (
    avaliar_melhor_modelo,
    buscar_hiperparametros,
    calcular_metricas,
    melhor_configuracao,
)


def _conjunto(n, semente):
    rng = np.random.default_rng(semente)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    df[TARGET_COLUMN] = (20 + 5 * df["mediaKills_hist"] + rng.normal(size=n)).round()
    return df


GRADE = {"max_depth": [1, 3, None], "min_samples_leaf": [1, 20]}


def test_metricas_em_kills():
    m = calcular_metricas(pd.Series([10, 20]), np.array([12, 16]))
    assert m["MAE"] == 3
    assert np.isclose(m["RMSE"], np.sqrt((4 + 16) / 2))


def test_busca_testa_todas_as_combinacoes_e_ordena_pela_validacao():
    busca = buscar_hiperparametros(_conjunto(300, 0), _conjunto(100, 1), GRADE)
    assert len(busca) == 6
    assert busca["MAE_validacao"].is_monotonic_increasing
    assert {"MAE_treino", "RMSE_validacao", "R2_validacao", "diferenca_MAE"} <= set(busca.columns)


def test_arvore_sem_limite_decora_o_treino():
    busca = buscar_hiperparametros(_conjunto(300, 0), _conjunto(100, 1), GRADE)
    sem_limite = busca[(busca["max_depth"] == "None") & (busca["min_samples_leaf"] == 1)].iloc[0]
    assert sem_limite["MAE_treino"] < 0.5
    assert sem_limite["diferenca_MAE"] > 0


def test_melhor_modelo_supera_prever_a_media_e_usa_feature_relevante():
    treino, validacao = _conjunto(500, 0), _conjunto(200, 1)
    configuracao = melhor_configuracao(buscar_hiperparametros(treino, validacao, GRADE))
    _, resumo, importancias = avaliar_melhor_modelo(treino, validacao, configuracao)
    mae_media = (validacao[TARGET_COLUMN] - treino[TARGET_COLUMN].mean()).abs().mean()
    assert resumo.loc[0, "MAE"] < mae_media
    assert resumo.loc[0, "partidas"] == 200
    assert importancias.loc[0, "feature"] == "mediaKills_hist"
    assert np.isclose(importancias["importancia"].sum(), 1)
