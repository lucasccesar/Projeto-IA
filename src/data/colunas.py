"""Colunas do dataset que estruturam o projeto e são usadas por vários módulos.

As colunas de estatísticas (dano, tiros, rounds...) ficam nos módulos que as
utilizam; aqui estão só as que identificam a linha, ordenam o histórico e
definem o alvo.
"""

PLAYER_COLUMN = "idPlayer"
MATCH_COLUMN = "idLobbyGame"
DATE_COLUMN = "dtCreatedAt"
TARGET_COLUMN = "qtKill"

# Cada linha do dataset é a participação de um jogador em uma partida.
KEY_COLUMNS = (PLAYER_COLUMN, MATCH_COLUMN)

# Identificam a linha e permitem a divisão por data. Não são entradas do modelo.
CONTEXT_COLUMNS = (PLAYER_COLUMN, MATCH_COLUMN, DATE_COLUMN)

# Ordem que define o que é "partida anterior": por jogador, da mais antiga para a
# mais recente. O ID da partida desempata partidas com a mesma data.
HISTORY_ORDER = (PLAYER_COLUMN, DATE_COLUMN, MATCH_COLUMN)
