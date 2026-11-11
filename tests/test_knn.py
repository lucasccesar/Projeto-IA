import numpy as np
import pandas as pd
import pytest

from src.data.dataset_final import FEATURE_COLUMNS, TARGET_COLUMN, separar_entradas_alvo
from src.evaluation.metricas import calcular_metricas
from src.models.knn import (
    buscar_hiperparametros,
    criar_modelo,
    melhor_configuracao,
    resumo_validacao,
)


def _conjunto(n, semente, deslocamento=0.0):
    rng = np.random.default_rng(semente)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    df += deslocamento
    df[TARGET_COLUMN] = 20 + 5 * df["mediaKills_hist"] + rng.normal(scale=0.5, size=n)
    return df


VALORES_K = [1, 5, 25]
DISTANCIAS = ["euclidean", "manhattan"]


def test_busca_testa_todas_as_combinacoes_e_ordena_pela_validacao():
    busca = buscar_hiperparametros(_conjunto(300, 0), _conjunto(100, 1), VALORES_K, DISTANCIAS)
    assert len(busca) == 6
    assert busca["MAE_validacao"].is_monotonic_increasing
    assert list(busca.columns) == ["k", "distancia", "MAE_validacao", "RMSE_validacao", "R2_validacao"]
    assert set(busca["k"]) == set(VALORES_K)
    assert set(busca["distancia"]) == set(DISTANCIAS)


def test_busca_rapida_equivale_a_treinar_um_knn_por_combinacao():
    treino, validacao = _conjunto(300, 0), _conjunto(100, 1)
    busca = buscar_hiperparametros(treino, validacao, VALORES_K, DISTANCIAS)
    X_treino, y_treino = separar_entradas_alvo(treino)
    X_val, y_val = separar_entradas_alvo(validacao)

    for linha in busca.itertuples():
        modelo = criar_modelo(linha.k, linha.distancia).fit(X_treino, y_treino)
        esperado = calcular_metricas(y_val, modelo.predict(X_val))
        assert np.isclose(linha.MAE_validacao, esperado["MAE"])
        assert np.isclose(linha.RMSE_validacao, esperado["RMSE"])
        assert np.isclose(linha.R2_validacao, esperado["R2"])


def test_padronizacao_usa_apenas_o_treino():
    treino = _conjunto(200, 0)
    X_treino, y_treino = separar_entradas_alvo(treino)
    modelo = criar_modelo(5, "euclidean").fit(X_treino, y_treino)
    escala = modelo[0]
    modelo.predict(_conjunto(50, 1, deslocamento=5)[list(FEATURE_COLUMNS)])
    np.testing.assert_allclose(escala.mean_, X_treino.mean().to_numpy())


def test_melhor_configuracao_e_resumo_usam_o_menor_mae():
    treino, validacao = _conjunto(500, 0), _conjunto(200, 1)
    busca = buscar_hiperparametros(treino, validacao, VALORES_K, DISTANCIAS)
    configuracao = melhor_configuracao(busca)
    resumo = resumo_validacao(busca, partidas=len(validacao))

    assert configuracao == {"k": busca.loc[0, "k"], "distancia": busca.loc[0, "distancia"]}
    assert configuracao["k"] > 1  # um único vizinho é ruidoso demais
    assert list(resumo.columns) == [
        "conjunto", "modelo", "k", "distancia", "partidas", "MAE", "RMSE", "R2",
    ]
    assert resumo.loc[0, "modelo"] == "knn"
    assert resumo.loc[0, "partidas"] == 200
    assert resumo.loc[0, "MAE"] == busca["MAE_validacao"].min()
    mae_media = (validacao[TARGET_COLUMN] - treino[TARGET_COLUMN].mean()).abs().mean()
    assert resumo.loc[0, "MAE"] < mae_media


def test_k_maior_que_o_treino_e_rejeitado():
    with pytest.raises(ValueError, match="maior K"):
        buscar_hiperparametros(_conjunto(10, 0), _conjunto(5, 1), [3, 11], ["euclidean"])
