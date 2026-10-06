import numpy as np
import pandas as pd

from src.preprocessing.feature_historico import (
    criar_features_historico,
    filtrar_minimo_partidas,
)


def _dados():
    """Jogador 10 com 3 partidas e jogador 20 com 1 partida."""
    return pd.DataFrame({
        'idLobbyGame':    [1, 2, 3, 4],
        'idPlayer':       [10, 10, 10, 20],
        'dtCreatedAt':    ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-01'],
        'qtKill':         [10, 20, 30, 5],
        'qtShots':        [200, 300, 400, 100],
        'vlDamage':       [2000, 3000, 4000, 1000],
        'qtHitHeadshot':  [8, 12, 16, 4],
        'qtRoundsPlayed': [20, 25, 30, 15],
        'flWinner':       [1, 0, 1, 0],
        'vlLevel':        [5, 6, 6, 3],
        'descMapName':    ['a', 'b', 'a', 'a'],
    })


def _jogador_10():
    df = criar_features_historico(_dados())
    return df[df['idPlayer'] == 10].reset_index(drop=True)


def test_primeira_partida_nao_tem_historico():
    df = criar_features_historico(_dados())
    primeiras = df[df['qtdAnteriores'] == 0]
    novas = [c for c in df.columns if c not in _dados().columns and c != 'qtdAnteriores']
    assert primeiras[novas].isna().all().all()


def test_medias_usam_so_partidas_anteriores():
    j = _jogador_10()
    assert j.loc[1, 'mediaKills_hist'] == 10
    assert j.loc[2, 'mediaKills_hist'] == 15
    assert j.loc[2, 'killsUltimaPartida'] == 20
    assert j.loc[2, 'mediaDano_hist'] == 2500
    assert j.loc[2, 'mediaHeadshots_hist'] == 10
    assert j.loc[2, 'mediaRounds_hist'] == 22.5
    assert j.loc[2, 'taxaVitorias_hist'] == 0.5


def test_razoes_somam_antes_de_dividir():
    j = _jogador_10()
    assert np.isclose(j.loc[2, 'tirosPorKill_hist'], (200 + 300) / (10 + 20))
    assert np.isclose(j.loc[2, 'killsPorRound_hist'], (10 + 20) / (20 + 25))


def test_desvio_precisa_de_duas_partidas_anteriores():
    j = _jogador_10()
    assert pd.isna(j.loc[1, 'desvioKills_20'])
    assert np.isclose(j.loc[2, 'desvioKills_20'], np.std([10, 20], ddof=1))


def test_nivel_e_o_da_partida_anterior():
    j = _jogador_10()
    assert pd.isna(j.loc[0, 'nivelAnterior'])
    assert j.loc[1, 'nivelAnterior'] == 5
    assert j.loc[2, 'nivelAnterior'] == 6


def test_media_no_mapa_considera_so_o_mesmo_mapa():
    j = _jogador_10()
    assert pd.isna(j.loc[1, 'mediaKillsMapa_hist'])   # 1ª vez no mapa b
    assert j.loc[2, 'mediaKillsMapa_hist'] == 10      # mapa a: só a 1ª partida


def test_mudar_a_partida_atual_nao_altera_suas_features():
    base = criar_features_historico(_dados())
    alterado = _dados()
    colunas = ['qtKill', 'qtShots', 'vlDamage', 'qtHitHeadshot', 'qtRoundsPlayed', 'flWinner', 'vlLevel']
    alterado.loc[2, colunas] = 999            # partida 3 do jogador 10
    novo = criar_features_historico(alterado)

    features = [c for c in base.columns if c not in _dados().columns]
    linha_base = base[base['idLobbyGame'] == 3][features].reset_index(drop=True)
    linha_nova = novo[novo['idLobbyGame'] == 3][features].reset_index(drop=True)
    assert linha_base.equals(linha_nova)


def test_filtro_do_minimo_de_partidas():
    dados = pd.DataFrame({
        'idLobbyGame': range(1, 8),
        'idPlayer': [1] * 7,
        'dtCreatedAt': pd.date_range('2020-01-01', periods=7),
        'qtKill': [10] * 7, 'qtShots': [100] * 7, 'vlDamage': [1000] * 7,
        'qtHitHeadshot': [5] * 7, 'qtRoundsPlayed': [20] * 7, 'flWinner': [1] * 7,
        'vlLevel': [5] * 7, 'descMapName': ['a'] * 7,
    })
    df = filtrar_minimo_partidas(criar_features_historico(dados))
    assert list(df['qtdAnteriores']) == [5, 6]
