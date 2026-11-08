import numpy as np
import pandas as pd

from src.data.dataset_final import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.regressao_linear import (
    avaliar_modelos,
    calcular_metricas,
    coeficientes,
    criar_modelo,
    resumo_validacao,
)


def _conjunto(n, semente, deslocamento=0.0):
    rng = np.random.default_rng(semente)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    df += deslocamento
    df[TARGET_COLUMN] = 20 + 3 * df["mediaKills_hist"] - 2 * df["nivelAnterior"] + rng.normal(scale=0.1, size=n)
    return df


def test_metricas_em_kills():
    m = calcular_metricas(pd.Series([10, 20]), np.array([12, 16]))
    assert m["MAE"] == 3
    assert np.isclose(m["RMSE"], np.sqrt((4 + 16) / 2))


def test_padronizacao_usa_apenas_o_treino():
    treino = _conjunto(200, 0)
    modelo = criar_modelo().fit(treino[list(FEATURE_COLUMNS)], treino[TARGET_COLUMN])
    escala = modelo[0]
    np.testing.assert_allclose(escala.mean_, treino[list(FEATURE_COLUMNS)].mean().to_numpy())
    validacao = _conjunto(200, 1, deslocamento=5)
    modelo.predict(validacao[list(FEATURE_COLUMNS)])
    np.testing.assert_allclose(escala.mean_, treino[list(FEATURE_COLUMNS)].mean().to_numpy())


def test_recupera_relacao_linear_conhecida():
    coef = coeficientes(_conjunto(2000, 0)).set_index("feature")["coeficiente"]
    assert coef.index[0] == "mediaKills_hist"
    assert np.isclose(coef["mediaKills_hist"], 3, atol=0.2)
    assert np.isclose(coef["nivelAnterior"], -2, atol=0.2)


def test_avalia_linear_e_cada_alfa_e_resume_os_dois_modelos():
    avaliacao = avaliar_modelos(_conjunto(300, 0), _conjunto(100, 1), alfas=[1, 1000])
    assert avaliacao["modelo"].tolist() == ["regressao_linear", "ridge", "ridge"]
    assert avaliacao["alfa"].tolist() == [0, 1, 1000]
    resumo = resumo_validacao(avaliacao, partidas=100)
    assert resumo["modelo"].tolist() == ["regressao_linear", "ridge"]
    assert resumo.loc[1, "alfa"] == 1
    assert (resumo["partidas"] == 100).all()
