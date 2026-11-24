import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.evaluation.metricas import calcular_metricas, metricas_treino_validacao
from src.evaluation.resultados import formatar_tabela, salvar_tabela


def test_metricas_em_kills():
    m = calcular_metricas(pd.Series([10, 20]), np.array([12, 16]))
    assert m["MAE"] == 3
    assert np.isclose(m["RMSE"], np.sqrt((4 + 16) / 2))
    assert np.isclose(m["R2"], 1 - (4 + 16) / 50)


def test_previsao_perfeita_e_previsao_da_media():
    y = np.array([5.0, 15.0, 25.0])
    assert calcular_metricas(y, y) == {"MAE": 0.0, "RMSE": 0.0, "R2": 1.0}
    assert calcular_metricas(y, np.full(3, y.mean()))["R2"] == 0.0


def test_metricas_de_treino_e_validacao_usam_os_conjuntos_certos():
    X_treino, y_treino = np.zeros((4, 1)), np.array([10.0, 10.0, 20.0, 20.0])
    X_val, y_val = np.zeros((2, 1)), np.array([15.0, 25.0])
    modelo = DummyRegressor(strategy="mean").fit(X_treino, y_treino)  # sempre prevê 15

    metricas = metricas_treino_validacao(modelo, X_treino, y_treino, X_val, y_val)

    assert list(metricas) == ["MAE_treino", "MAE_validacao", "RMSE_validacao", "R2_validacao"]
    assert metricas["MAE_treino"] == 5
    assert metricas["MAE_validacao"] == 5
    assert np.isclose(metricas["RMSE_validacao"], np.sqrt(50))


def test_salva_csv_com_seis_casas_e_exibe_com_tres(tmp_path):
    tabela = pd.DataFrame({"modelo": ["a"], "MAE": [1.23456789]})

    caminho = salvar_tabela(tabela, "metricas.csv", pasta=tmp_path / "metrics")

    assert caminho == tmp_path / "metrics" / "metricas.csv"
    assert caminho.read_text() == "modelo,MAE\na,1.234568\n"
    assert formatar_tabela(tabela).split()[-1] == "1.235"
