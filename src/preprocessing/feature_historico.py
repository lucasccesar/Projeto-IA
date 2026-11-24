"""Features do histórico do jogador: o que se sabe sobre ele antes da partida começar.

Toda feature usa somente partidas anteriores do mesmo jogador. A partida da
própria linha nunca entra no cálculo, para não vazar o resultado que se quer prever.
"""

import numpy as np
import pandas as pd

from src.data.colunas import (
    DATE_COLUMN,
    HISTORY_ORDER,
    MATCH_COLUMN,
    PLAYER_COLUMN,
    TARGET_COLUMN,
)

JANELA_CURTA = 5
JANELA_LONGA = 20
MIN_PARTIDAS = 5
LEVEL_COLUMN = "vlLevel"

# {sufixo da feature: quantidade de partidas anteriores}. None usa todo o histórico.
JANELAS = {"5": JANELA_CURTA, "20": JANELA_LONGA, "hist": None}

# {nome da feature: coluna do dataset} -> média nas partidas anteriores
MEDIAS = {
    "mediaKills": TARGET_COLUMN,
    "mediaDano": "vlDamage",
    "mediaHeadshots": "qtHitHeadshot",
    "mediaRounds": "qtRoundsPlayed",
    "taxaVitorias": "flWinner",
}

# {nome da feature: (numerador, denominador)} -> soma anterior / soma anterior
RAZOES = {
    "tirosPorKill": ("qtShots", TARGET_COLUMN),
    "killsPorRound": (TARGET_COLUMN, "qtRoundsPlayed"),
}

DESVIO_KILLS = f"desvioKills_{JANELA_LONGA}"

# Colunas do dataset bruto necessárias para calcular as features. São derivadas
# das definições acima para nunca ficarem desatualizadas.
REQUIRED_FEATURE_COLUMNS = tuple(dict.fromkeys([
    PLAYER_COLUMN,
    MATCH_COLUMN,
    DATE_COLUMN,
    *MEDIAS.values(),
    *(coluna for par in RAZOES.values() for coluna in par),
    LEVEL_COLUMN,
]))

# Entradas do modelo, na ordem em que aparecem no dataset final.
FEATURE_COLUMNS = (
    "qtdAnteriores",
    "killsUltimaPartida",
    "nivelAnterior",
    *(f"{nome}_{sufixo}" for nome in MEDIAS for sufixo in JANELAS),
    *(f"{nome}_{sufixo}" for nome in RAZOES for sufixo in JANELAS),
    DESVIO_KILLS,
)


def agregar_partidas_anteriores(
    df: pd.DataFrame,
    coluna: str,
    operacao: str,
    janela: int | None = None,
    min_periods: int = 1,
) -> pd.Series:
    """Aplica uma operação às partidas anteriores do jogador (a atual não entra).

    `operacao` é o nome de uma agregação do pandas: "mean", "sum", "std", "min" ou "max".
    janela=None usa todo o histórico; janela=N usa as N partidas anteriores mais recentes.
    O resultado fica vazio (NaN) enquanto o jogador tem menos de `min_periods` partidas
    anteriores. O DataFrame precisa estar ordenado por `ordenar_por_tempo`.
    """
    def calcular(valores: pd.Series) -> pd.Series:
        anteriores = valores.shift(1)  # tira a partida atual
        if janela is None:
            historico = anteriores.expanding(min_periods=min_periods)
        else:
            historico = anteriores.rolling(janela, min_periods=min_periods)
        return getattr(historico, operacao)()

    return df.groupby(PLAYER_COLUMN)[coluna].transform(calcular)


def media_historica(df: pd.DataFrame, coluna: str, janela: int | None = None) -> pd.Series:
    """Média da coluna nas partidas anteriores do jogador (a atual não entra).

    janela=None usa todo o histórico; janela=N usa as N partidas anteriores mais recentes.
    """
    return agregar_partidas_anteriores(df, coluna, "mean", janela)


def desvio_historico(df: pd.DataFrame, coluna: str, janela: int) -> pd.Series:
    """Desvio-padrão da coluna nas últimas `janela` partidas anteriores (precisa de 2 ou mais)."""
    return agregar_partidas_anteriores(df, coluna, "std", janela, min_periods=2)


def razao_historica(
    df: pd.DataFrame, numerador: str, denominador: str, janela: int | None = None
) -> pd.Series:
    """Soma do numerador / soma do denominador nas partidas anteriores do jogador.

    Somar antes de dividir evita divisão por zero em partidas isoladas
    (por exemplo, partidas com 0 kills).
    """
    soma_numerador = agregar_partidas_anteriores(df, numerador, "sum", janela)
    soma_denominador = agregar_partidas_anteriores(df, denominador, "sum", janela)
    return soma_numerador / soma_denominador.replace(0, np.nan)


def ordenar_por_tempo(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena por jogador e data, sem alterar o DataFrame recebido.

    O histórico depende dessa ordem.
    """
    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df.sort_values(list(HISTORY_ORDER)).reset_index(drop=True)


def criar_features_historico(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as informações do histórico do jogador. Recebe o dataset bruto ou limpo.

    Todas as colunas novas usam só partidas anteriores à da linha. Na primeira
    partida de cada jogador elas ficam vazias (NaN), pois não há histórico.
    """
    df = ordenar_por_tempo(df)
    por_jogador = df.groupby(PLAYER_COLUMN)

    # Contexto do jogador
    df["qtdAnteriores"] = por_jogador.cumcount()
    df["killsUltimaPartida"] = por_jogador[TARGET_COLUMN].shift(1)
    df["nivelAnterior"] = por_jogador[LEVEL_COLUMN].shift(1)

    # Médias nas janelas: kills, dano, headshots, rounds e vitórias
    for nome, coluna in MEDIAS.items():
        for sufixo, janela in JANELAS.items():
            df[f"{nome}_{sufixo}"] = media_historica(df, coluna, janela)

    # Razões nas janelas: tiros por kill e kills por round
    for nome, (numerador, denominador) in RAZOES.items():
        for sufixo, janela in JANELAS.items():
            df[f"{nome}_{sufixo}"] = razao_historica(df, numerador, denominador, janela)

    # Consistência: quanto as kills variam nas partidas recentes
    df[DESVIO_KILLS] = desvio_historico(df, TARGET_COLUMN, JANELA_LONGA)

    return df


def filtrar_minimo_partidas(df: pd.DataFrame, minimo: int = MIN_PARTIDAS) -> pd.DataFrame:
    """Mantém só as linhas em que o jogador já tinha pelo menos `minimo` partidas anteriores."""
    return df[df["qtdAnteriores"] >= minimo]
