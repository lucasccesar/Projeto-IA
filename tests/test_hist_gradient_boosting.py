import numpy as np
import pandas as pd

from src.data.dataset_final import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.hist_gradient_boosting import (
    LOSSES,
    buscar_hiperparametros,
    calcular_metricas,
    combinacoes,
    criar_modelo,
    melhor_configuracao,
    resumo_validacao,
)


def _conjunto(n, semente):
    rng = np.random.default_rng(semente)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    df[TARGET_COLUMN] = (20 + 5 * df["mediaKills_hist"] + rng.normal(size=n)).round()
    return df


GRADE = {
    "learning_rate": [0.1],
    "max_iter": [5, 50],
    "max_leaf_nodes": [7],
    "min_samples_leaf": [10],
    "l2_regularization": [0.0],
}


def test_metricas_em_kills():
    m = calcular_metricas(pd.Series([10, 20]), np.array([12, 16]))
    assert m["MAE"] == 3
    assert np.isclose(m["RMSE"], np.sqrt((4 + 16) / 2))


def test_grade_padrao_tem_as_duas_losses():
    configuracoes = combinacoes()
    assert {c["loss"] for c in configuracoes} == {"squared_error", "absolute_error"}
    assert len(configuracoes) == 2 * 32


def test_modelo_sem_early_stopping_e_com_semente_fixa():
    modelo = criar_modelo(loss="absolute_error")
    assert modelo.early_stopping is False
    assert modelo.random_state == 42
    assert modelo.loss == "absolute_error"


def test_busca_avalia_todas_as_configuracoes_das_duas_losses():
    busca = buscar_hiperparametros(_conjunto(300, 0), _conjunto(100, 1), GRADE)
    assert len(busca) == 2 * len(LOSSES)
    assert set(busca["loss"]) == set(LOSSES)
    assert busca["MAE_validacao"].is_monotonic_increasing
    assert (busca["partidas_treino"] == 300).all()
    assert (busca["partidas_validacao"] == 100).all()
    assert {"MAE_treino", "RMSE_validacao", "R2_validacao", "diferenca_MAE"} <= set(busca.columns)


def test_selecao_pelo_menor_mae_de_validacao():
    treino, validacao = _conjunto(500, 0), _conjunto(200, 1)
    busca = buscar_hiperparametros(treino, validacao, GRADE)
    escolhida = melhor_configuracao(busca)
    assert escolhida["loss"] == busca.loc[busca["MAE_validacao"].idxmin(), "loss"]

    resumo = resumo_validacao(busca)
    assert len(resumo) == len(LOSSES)
    assert resumo["selecionado"].sum() == 1
    selecionada = resumo[resumo["selecionado"]].iloc[0]
    assert np.isclose(selecionada["MAE"], busca["MAE_validacao"].min())
    assert selecionada["max_iter"] == escolhida["max_iter"]
    assert (resumo["partidas"] == 200).all()

    mae_media = (validacao[TARGET_COLUMN] - treino[TARGET_COLUMN].mean()).abs().mean()
    assert selecionada["MAE"] < mae_media


def test_melhor_configuracao_por_loss():
    busca = buscar_hiperparametros(_conjunto(300, 0), _conjunto(100, 1), GRADE)
    for loss in LOSSES:
        assert melhor_configuracao(busca, loss)["loss"] == loss
