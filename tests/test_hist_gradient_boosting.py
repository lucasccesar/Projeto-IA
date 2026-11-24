import numpy as np
import pandas as pd
import pytest

from src.data.dataset_final import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.hist_gradient_boosting import (
    BASELINE,
    LOSSES,
    MODELO,
    avaliar_no_teste,
    buscar_hiperparametros,
    combinacoes,
    configuracao_selecionada,
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
    assert {"MAE_treino", "RMSE_validacao", "R2_validacao", "diferenca_MAE"} <= set(busca.columns)


def test_resumo_tem_a_melhor_configuracao_de_cada_loss_no_formato_padrao():
    treino, validacao = _conjunto(500, 0), _conjunto(200, 1)
    busca = buscar_hiperparametros(treino, validacao, GRADE)
    resumo = resumo_validacao(busca, len(validacao))

    assert list(resumo.columns[:2]) == ["conjunto", "modelo"]
    assert list(resumo.columns[-4:]) == ["partidas", "MAE", "RMSE", "R2"]
    assert len(resumo) == len(LOSSES)
    assert set(resumo["loss"]) == set(LOSSES)
    assert (resumo["conjunto"] == "validacao").all()
    assert (resumo["modelo"] == MODELO).all()
    assert (resumo["partidas"] == 200).all()
    assert resumo["MAE"].is_monotonic_increasing

    escolhida = resumo.iloc[0]
    assert np.isclose(escolhida["MAE"], busca["MAE_validacao"].min())
    mae_media = (validacao[TARGET_COLUMN] - treino[TARGET_COLUMN].mean()).abs().mean()
    assert escolhida["MAE"] < mae_media


def test_configuracao_selecionada_e_a_de_menor_mae_na_validacao():
    busca = buscar_hiperparametros(_conjunto(500, 0), _conjunto(200, 1), GRADE)
    resumo = resumo_validacao(busca, 200)
    # A ordem das linhas não importa: vale a menor MAE.
    assert configuracao_selecionada(resumo[::-1]) == melhor_configuracao(busca)


def test_configuracao_selecionada_exige_resumo_preenchido():
    with pytest.raises(ValueError):
        configuracao_selecionada(pd.DataFrame(columns=["loss", "MAE"]))


def test_avaliacao_no_teste_usa_a_configuracao_escolhida_e_a_baseline():
    treino, validacao, teste = _conjunto(500, 0), _conjunto(200, 1), _conjunto(150, 2)
    configuracao = melhor_configuracao(buscar_hiperparametros(treino, validacao, GRADE))

    resultado = avaliar_no_teste(treino, teste, configuracao)
    assert list(resultado.columns[:2]) == ["conjunto", "modelo"]
    assert list(resultado.columns[-4:]) == ["partidas", "MAE", "RMSE", "R2"]
    assert list(resultado["modelo"]) == [BASELINE, MODELO]
    assert (resultado["conjunto"] == "teste").all()
    assert (resultado["partidas"] == 150).all()

    baseline, modelo = resultado.iloc[0], resultado.iloc[1]
    mae_baseline = (teste[TARGET_COLUMN] - teste["mediaKills_hist"]).abs().mean()
    assert np.isclose(baseline["MAE"], mae_baseline)
    assert pd.isna(baseline["loss"])
    assert modelo["loss"] == configuracao["loss"]
    assert modelo["max_iter"] == configuracao["max_iter"]


def test_melhor_configuracao_por_loss():
    busca = buscar_hiperparametros(_conjunto(300, 0), _conjunto(100, 1), GRADE)
    for loss in LOSSES:
        assert melhor_configuracao(busca, loss)["loss"] == loss
