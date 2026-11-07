import pandas as pd
import pytest

from src.preprocessing.separacao_dados import dividir_por_tempo, verificar_ordem_temporal


def _dados(n=100):
    return pd.DataFrame({
        'idPlayer': [i % 5 for i in range(n)],
        'dtCreatedAt': pd.date_range('2020-01-01', periods=n, freq='D'),
        'qtKill': range(n),
    }).sample(frac=1, random_state=0)   # embaralhado, para testar a ordenação


def test_proporcoes_e_total():
    treino, validacao, teste = dividir_por_tempo(_dados())
    assert len(treino) + len(validacao) + len(teste) == 100
    assert len(treino) == 70
    assert len(validacao) == 15
    assert len(teste) == 15


def test_ordem_temporal_sem_sobreposicao():
    treino, validacao, teste = dividir_por_tempo(_dados())
    verificar_ordem_temporal(treino, validacao, teste)
    assert treino['dtCreatedAt'].max() < validacao['dtCreatedAt'].min()
    assert validacao['dtCreatedAt'].max() < teste['dtCreatedAt'].min()


def test_nenhuma_linha_repetida_ou_perdida():
    dados = _dados()
    treino, validacao, teste = dividir_por_tempo(dados)
    todas = pd.concat([treino, validacao, teste])
    assert len(todas) == len(dados)
    assert not todas.duplicated().any()


def test_mesma_data_fica_no_mesmo_conjunto():
    dados = pd.DataFrame({
        'idPlayer': range(10),
        'dtCreatedAt': pd.to_datetime(['2020-01-01'] * 5 + ['2020-01-02'] * 5),
    })
    treino, validacao, teste = dividir_por_tempo(dados, 0.5, 0.2)
    for conjunto in (treino, validacao, teste):
        assert conjunto['dtCreatedAt'].nunique() <= 1


def test_proporcoes_invalidas():
    with pytest.raises(ValueError):
        dividir_por_tempo(_dados(), 0.8, 0.3)
