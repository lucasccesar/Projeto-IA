import pandas as pd

from src.preprocessing.feature_historico import criar_features_historico


def _dados():
    return pd.DataFrame({
        'idLobbyGame': [1, 2, 3, 4],
        'idPlayer': [10, 10, 10, 20],
        'qtKill': [10, 20, 30, 5],
        'dtCreatedAt': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-01'],
    })


def test_primeira_partida_nao_tem_historico():
    df = criar_features_historico(_dados())
    primeiras = df[df['qtdAnteriores'] == 0]
    assert primeiras['mediaKills_hist'].isna().all()
    assert primeiras['killsUltimaPartida'].isna().all()


def test_historico_usa_so_partidas_anteriores():
    df = criar_features_historico(_dados())
    jogador = df[df['idPlayer'] == 10].reset_index(drop=True)
    assert jogador.loc[1, 'mediaKills_hist'] == 10      # só a 1ª partida
    assert jogador.loc[2, 'mediaKills_hist'] == 15      # média de 10 e 20
    assert jogador.loc[2, 'killsUltimaPartida'] == 20

