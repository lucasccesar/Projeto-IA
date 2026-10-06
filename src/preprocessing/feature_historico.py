import numpy as np
import pandas as pd

JANELA_CURTA = 5
JANELA_LONGA = 20
MIN_PARTIDAS = 5

# Janelas usadas nas médias: últimas 5, últimas 20 e todo o histórico (None)
JANELAS = [('5', JANELA_CURTA), ('20', JANELA_LONGA), ('hist', None)]

# {nome da feature: coluna do dataset} -> média histórica
MEDIAS = {
    'mediaKills': 'qtKill',
    'mediaDano': 'vlDamage',
    'mediaHeadshots': 'qtHitHeadshot',
    'mediaRounds': 'qtRoundsPlayed',
    'taxaVitorias': 'flWinner',
}

# {nome da feature: (numerador, denominador)} -> soma anterior / soma anterior
RAZOES = {
    'tirosPorKill': ('qtShots', 'qtKill'),
    'killsPorRound': ('qtKill', 'qtRoundsPlayed'),
}


def media_historica(df, coluna, janela=None):
    """Média da coluna nas partidas anteriores do jogador (a atual não entra).

    janela=None usa todo o histórico; janela=N usa as N partidas anteriores mais recentes.
    """
    g = df.groupby('idPlayer')[coluna]
    if janela is None:
        return g.transform(lambda s: s.shift(1).expanding().mean())
    return g.transform(lambda s: s.shift(1).rolling(janela, min_periods=1).mean())


def desvio_historico(df, coluna, janela):
    """Desvio-padrão da coluna nas últimas `janela` partidas anteriores (precisa de 2 ou mais)."""
    g = df.groupby('idPlayer')[coluna]
    return g.transform(lambda s: s.shift(1).rolling(janela, min_periods=2).std())


def razao_historica(df, numerador, denominador, janela=None):
    """Soma do numerador / soma do denominador nas partidas anteriores do jogador.

    Somar antes de dividir evita divisão por zero em partidas isoladas
    (por exemplo, partidas com 0 kills).
    """
    g = df.groupby('idPlayer')

    def soma(coluna):
        if janela is None:
            return g[coluna].transform(lambda s: s.shift(1).expanding().sum())
        return g[coluna].transform(lambda s: s.shift(1).rolling(janela, min_periods=1).sum())

    return soma(numerador) / soma(denominador).replace(0, np.nan)


def media_historica_no_mapa(df, coluna):
    """Média da coluna nas partidas anteriores do jogador no mesmo mapa da partida."""
    g = df.groupby(['idPlayer', 'descMapName'])[coluna]
    return g.transform(lambda s: s.shift(1).expanding().mean())


def ordenar_por_tempo(df):
    """Ordena por jogador e data. O histórico depende dessa ordem."""
    df = df.copy()
    df['dtCreatedAt'] = pd.to_datetime(df['dtCreatedAt'])
    return df.sort_values(['idPlayer', 'dtCreatedAt', 'idLobbyGame']).reset_index(drop=True)


def criar_features_historico(df):
    """Cria as informações do histórico do jogador. Recebe o dataset bruto.

    Todas as colunas novas usam só partidas anteriores à da linha. Na primeira
    partida de cada jogador elas ficam vazias (NaN), pois não há histórico.
    """
    df = ordenar_por_tempo(df)
    por_jogador = df.groupby('idPlayer')

    # Contexto do jogador
    df['qtdAnteriores'] = por_jogador.cumcount()
    df['killsUltimaPartida'] = por_jogador['qtKill'].shift(1)
    df['nivelAnterior'] = por_jogador['vlLevel'].shift(1)

    # Médias nas janelas: kills, dano, headshots, rounds e vitórias
    for nome, coluna in MEDIAS.items():
        for sufixo, janela in JANELAS:
            df[f'{nome}_{sufixo}'] = media_historica(df, coluna, janela)

    # Razões nas janelas: tiros por kill e kills por round
    for nome, (numerador, denominador) in RAZOES.items():
        for sufixo, janela in JANELAS:
            df[f'{nome}_{sufixo}'] = razao_historica(df, numerador, denominador, janela)

    # Consistência e mapa
    df['desvioKills_20'] = desvio_historico(df, 'qtKill', JANELA_LONGA)
    df['mediaKillsMapa_hist'] = media_historica_no_mapa(df, 'qtKill')

    return df


def filtrar_minimo_partidas(df, minimo=MIN_PARTIDAS):
    """Mantém só as linhas em que o jogador já tinha pelo menos `minimo` partidas anteriores."""
    return df[df['qtdAnteriores'] >= minimo]
