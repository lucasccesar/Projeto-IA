"""Baselines cronológicas para prever kills a partir do histórico do jogador.

São regras simples, sem treinamento, que servem de referência: um modelo só
vale a pena se errar menos que elas nas mesmas partidas.
"""

import pandas as pd

from src.data.colunas import CONTEXT_COLUMNS, DATE_COLUMN, KEY_COLUMNS, PLAYER_COLUMN, TARGET_COLUMN
from src.data.dataset_final import carregar_dados_brutos, preparar_dataset_final
from src.evaluation.metricas import calcular_metricas
from src.evaluation.resultados import formatar_tabela, salvar_tabela
from src.preprocessing.feature_historico import (
    JANELA_LONGA,
    MIN_PARTIDAS,
    agregar_partidas_anteriores,
    ordenar_por_tempo,
)
from src.preprocessing.separacao_dados import dividir_por_tempo

ARQUIVO_RESUMO = "baseline_kills.csv"

# {nome da regra: (operação, janela)}. Cada regra resume as kills das partidas
# anteriores do jogador; janela=None usa todo o histórico.
REGRAS = {
    "media": ("mean", None),
    "media_20": ("mean", JANELA_LONGA),
    "minimo": ("min", None),
    "maximo": ("max", None),
}


def prever_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna previsões baseadas apenas nas partidas anteriores de cada jogador.

    Adiciona `qtdAnteriores` e uma coluna `previsao_<regra>` para cada regra de REGRAS.
    """
    ausentes = {*CONTEXT_COLUMNS, TARGET_COLUMN} - set(df.columns)
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(sorted(ausentes))}")

    partidas = ordenar_por_tempo(df)
    partidas["qtdAnteriores"] = partidas.groupby(PLAYER_COLUMN).cumcount()
    for nome, (operacao, janela) in REGRAS.items():
        partidas[f"previsao_{nome}"] = agregar_partidas_anteriores(
            partidas, TARGET_COLUMN, operacao, janela
        )
    return partidas


def avaliar_baselines(
    previsoes: pd.DataFrame, dataset_final: pd.DataFrame, minimo_partidas: int = MIN_PARTIDAS
) -> pd.DataFrame:
    """Avalia as regras nas mesmas partidas de validação do dataset final."""
    colunas = [f"previsao_{nome}" for nome in REGRAS]
    _, validacao, _ = dividir_por_tempo(dataset_final)
    if validacao.empty:
        raise ValueError("Não há partidas no conjunto de validação.")
    if (validacao["qtdAnteriores"] < minimo_partidas).any():
        raise ValueError("A validação contém jogadores sem histórico suficiente.")

    # Associa cada partida da validação à sua previsão, pela chave jogador + partida.
    chaves = list(KEY_COLUMNS)
    origem = previsoes[[*chaves, DATE_COLUMN, TARGET_COLUMN, *colunas]].rename(
        columns={DATE_COLUMN: "data_origem", TARGET_COLUMN: "kills_origem"}
    )
    avaliadas = validacao[[*chaves, DATE_COLUMN, TARGET_COLUMN]].merge(
        origem, on=chaves, how="left", validate="one_to_one", indicator=True
    )
    if (avaliadas["_merge"] != "both").any() or avaliadas[colunas].isna().any().any():
        raise ValueError("Faltam previsões para partidas do conjunto de validação.")
    if not (
        avaliadas[DATE_COLUMN].eq(avaliadas["data_origem"]).all()
        and avaliadas[TARGET_COLUMN].eq(avaliadas["kills_origem"]).all()
    ):
        raise ValueError("O dataset final e as previsões têm partidas divergentes.")

    kills = avaliadas[TARGET_COLUMN].to_numpy(dtype=float)
    return pd.DataFrame([
        {
            "conjunto": "validacao",
            "regra": nome,
            "partidas": len(avaliadas),
            "minimo_partidas_anteriores": minimo_partidas,
            **calcular_metricas(kills, avaliadas[f"previsao_{nome}"].to_numpy(dtype=float)),
        }
        for nome in REGRAS
    ])


def main() -> None:
    dados = carregar_dados_brutos()
    dataset_final = preparar_dataset_final(dados)
    metricas = avaliar_baselines(prever_baselines(dados), dataset_final)
    caminho = salvar_tabela(metricas, ARQUIVO_RESUMO)
    print(formatar_tabela(metricas))
    print(f"Resultados salvos em {caminho}")


if __name__ == "__main__":
    main()
