import numpy as np
import pandas as pd
import pytest

from src.models.baseline import avaliar_baselines, prever_baselines


def _partidas():
    return pd.DataFrame({
        "idLobbyGame": [6, 1, 8, 4, 2, 7, 3, 5],
        "idPlayer": [1, 1, 2, 1, 1, 2, 1, 1],
        "dtCreatedAt": [
            "2022-01-06", "2022-01-01", "2022-01-02", "2022-01-04",
            "2022-01-02", "2022-01-01", "2022-01-03", "2022-01-05",
        ],
        "qtKill": [30, 10, 100, 20, 0, 50, 5, 15],
    })


def test_previsoes_usam_apenas_historico_do_mesmo_jogador():
    resultado = prever_baselines(_partidas())
    jogador = resultado.loc[resultado["idPlayer"] == 1].reset_index(drop=True)
    assert jogador["qtdAnteriores"].tolist() == list(range(6))
    assert jogador.loc[0, ["previsao_media", "previsao_media_20", "previsao_minimo", "previsao_maximo"]].isna().all()
    assert jogador.loc[5, "previsao_media"] == 10
    assert jogador.loc[5, "previsao_media_20"] == 10
    assert jogador.loc[5, "previsao_minimo"] == 0
    assert jogador.loc[5, "previsao_maximo"] == 20
    outro = resultado.loc[resultado["idPlayer"] == 2].reset_index(drop=True)
    assert outro.loc[1, "previsao_media"] == 50


def test_partida_atual_nao_muda_sua_previsao():
    alterado = _partidas()
    alterado.loc[alterado["idLobbyGame"] == 6, "qtKill"] = 999
    colunas = ["previsao_media", "previsao_media_20", "previsao_minimo", "previsao_maximo"]
    original = prever_baselines(_partidas()).set_index("idLobbyGame").loc[6, colunas]
    novo = prever_baselines(alterado).set_index("idLobbyGame").loc[6, colunas]
    pd.testing.assert_series_equal(original, novo)


def test_media_20_usa_somente_as_20_partidas_anteriores():
    partidas = pd.DataFrame({
        "idLobbyGame": range(1, 23),
        "idPlayer": [1] * 22,
        "dtCreatedAt": pd.date_range("2022-01-01", periods=22),
        "qtKill": range(22),
    })
    resultado = prever_baselines(partidas)
    assert resultado.loc[21, "previsao_media_20"] == np.mean(range(1, 21))
    assert resultado.loc[21, "previsao_media"] == np.mean(range(21))


def test_metricas_comparam_as_quatro_regras_nas_mesmas_linhas():
    partidas = pd.DataFrame({
        "idLobbyGame": range(1, 21),
        "idPlayer": [1] * 20,
        "dtCreatedAt": pd.date_range("2022-01-01", periods=20),
        "qtKill": range(20),
    })
    metricas = avaliar_baselines(prever_baselines(partidas))
    assert metricas["conjunto"].tolist() == ["validacao"] * 4
    assert metricas["regra"].tolist() == ["media", "media_20", "minimo", "maximo"]
    assert metricas["partidas"].tolist() == [2] * 4
    np.testing.assert_allclose(metricas["MAE"], [8.25, 8.25, 15.5, 1])
    np.testing.assert_allclose(metricas["RMSE"], [np.sqrt((8**2 + 8.5**2) / 2)] * 2 + [np.sqrt((15**2 + 16**2) / 2), 1])


def test_sem_partidas_elegiveis_falha_com_mensagem():
    with pytest.raises(ValueError, match="Não há partidas elegíveis"):
        avaliar_baselines(prever_baselines(_partidas()), minimo_partidas=10)
