import pandas as pd

JANELA_CURTA = 5
JANELA_LONGA = 20
MIN_PARTIDAS = 5

def media_historica(df, coluna, janela=None):
    """Média da coluna nas partidas anteriores do jogador (a atual não entra).

    janela=None usa todo o histórico; janela=N usa as N partidas anteriores mais recentes.
    """
    g = df.groupby('idPlayer')[coluna]
    if janela is None:
        return g.transform(lambda s: s.shift(1).expanding().mean())
    return g.transform(lambda s: s.shift(1).rolling(janela, min_periods=1).mean())


def ordenar_por_tempo(df):
    """Ordena por jogador e data. O histórico depende dessa ordem."""
    df = df.copy()
    df['dtCreatedAt'] = pd.to_datetime(df['dtCreatedAt'])
    return df.sort_values(['idPlayer', 'dtCreatedAt', 'idLobbyGame']).reset_index(drop=True)


def criar_features_historico(df):
    """Cria as informações do histórico do jogador. Recebe o dataset bruto."""
    df = ordenar_por_tempo(df)

    df['qtdAnteriores'] = df.groupby('idPlayer').cumcount()
    df['killsUltimaPartida'] = df.groupby('idPlayer')['qtKill'].shift(1)
    df['mediaKills_5'] = media_historica(df, 'qtKill', JANELA_CURTA)
    df['mediaKills_20'] = media_historica(df, 'qtKill', JANELA_LONGA)
    df['mediaKills_hist'] = media_historica(df, 'qtKill')

    return df
