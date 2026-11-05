"""Baselines cronológicas para prever kills a partir do histórico do jogador."""

from pathlib import Path

import pandas as pd

from src.preprocessing.feature_historico import MIN_PARTIDAS, ordenar_por_tempo


REGRAS = {
    "media": "mean",
    "minimo": "min",
    "maximo": "max",
}


def prever_baselines(df):
    """Retorna previsões baseadas apenas nas partidas anteriores de cada jogador."""
    obrigatorias = {"idPlayer", "idLobbyGame", "dtCreatedAt", "qtKill"}
    ausentes = obrigatorias - set(df.columns)
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(sorted(ausentes))}")

    partidas = ordenar_por_tempo(df)
    partidas["qtdAnteriores"] = partidas.groupby("idPlayer").cumcount()
    kills_anteriores = partidas.groupby("idPlayer")["qtKill"].transform(
        lambda s: s.shift(1)
    )

    for nome, operacao in REGRAS.items():
        partidas[f"previsao_{nome}"] = kills_anteriores.groupby(
            partidas["idPlayer"]
        ).transform(lambda s: getattr(s.expanding(), operacao)())

    return partidas


def avaliar_baselines(previsoes, minimo_partidas=MIN_PARTIDAS):
    """Calcula MAE e RMSE nas mesmas partidas elegíveis para todas as regras."""
    colunas = [f"previsao_{nome}" for nome in REGRAS]
    elegiveis = previsoes.loc[
        (previsoes["qtdAnteriores"] >= minimo_partidas)
        & previsoes[["qtKill", *colunas]].notna().all(axis=1)
    ]
    if elegiveis.empty:
        raise ValueError("Não há partidas elegíveis para avaliar o baseline.")

    resultados = []
    for nome in REGRAS:
        erros = elegiveis["qtKill"] - elegiveis[f"previsao_{nome}"]
        resultados.append({
            "regra": nome,
            "partidas": len(elegiveis),
            "minimo_partidas_anteriores": minimo_partidas,
            "MAE": erros.abs().mean(),
            "RMSE": (erros.pow(2).mean()) ** 0.5,
        })
    return pd.DataFrame(resultados)


def main():
    raiz = Path(__file__).resolve().parents[2]
    entrada = raiz / "data/raw/tb_lobby_stats_player.csv"
    saida = raiz / "results/metrics/baseline_kills.csv"
    dados = pd.read_csv(entrada)
    metricas = avaliar_baselines(prever_baselines(dados))
    saida.parent.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(saida, index=False, float_format="%.6f")
    print(metricas.to_string(index=False, float_format=lambda valor: f"{valor:.3f}"))
    print(f"Resultados salvos em {saida}")


if __name__ == "__main__":
    main()
