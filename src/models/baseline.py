"""Baselines cronológicas para prever kills a partir do histórico do jogador."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score

from src.data.dataset_final import preparar_dataset_final
from src.preprocessing.feature_historico import JANELA_LONGA, MIN_PARTIDAS, ordenar_por_tempo
from src.preprocessing.separacao_dados import dividir_por_tempo


REGRAS = {
    "media": ("mean", None),
    "media_20": ("mean", JANELA_LONGA),
    "minimo": ("min", None),
    "maximo": ("max", None),
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

    for nome, (operacao, janela) in REGRAS.items():
        partidas[f"previsao_{nome}"] = kills_anteriores.groupby(
            partidas["idPlayer"]
        ).transform(
            lambda s: getattr(
                s.expanding() if janela is None else s.rolling(janela, min_periods=1),
                operacao,
            )()
        )

    return partidas


def avaliar_baselines(previsoes, dataset_final, minimo_partidas=MIN_PARTIDAS):
    """Avalia as regras nas mesmas partidas de validação do dataset final."""
    colunas = [f"previsao_{nome}" for nome in REGRAS]
    _, validacao, _ = dividir_por_tempo(dataset_final)
    if validacao.empty:
        raise ValueError("Não há partidas no conjunto de validação.")
    if (validacao["qtdAnteriores"] < minimo_partidas).any():
        raise ValueError("A validação contém jogadores sem histórico suficiente.")

    chaves = ["idPlayer", "idLobbyGame"]
    origem = previsoes[chaves + ["dtCreatedAt", "qtKill", *colunas]].rename(
        columns={"dtCreatedAt": "data_origem", "qtKill": "kills_origem"}
    )
    avaliadas = validacao[chaves + ["dtCreatedAt", "qtKill"]].merge(
        origem, on=chaves, how="left", validate="one_to_one", indicator=True
    )
    if (avaliadas["_merge"] != "both").any() or avaliadas[colunas].isna().any().any():
        raise ValueError("Faltam previsões para partidas do conjunto de validação.")
    if not (
        avaliadas["dtCreatedAt"].eq(avaliadas["data_origem"]).all()
        and avaliadas["qtKill"].eq(avaliadas["kills_origem"]).all()
    ):
        raise ValueError("O dataset final e as previsões têm partidas divergentes.")

    resultados = []
    for nome in REGRAS:
        previsao = avaliadas[f"previsao_{nome}"]
        erros = avaliadas["qtKill"] - previsao
        resultados.append({
            "conjunto": "validacao",
            "regra": nome,
            "partidas": len(avaliadas),
            "minimo_partidas_anteriores": minimo_partidas,
            "MAE": erros.abs().mean(),
            "RMSE": (erros.pow(2).mean()) ** 0.5,
            "R2": r2_score(avaliadas["qtKill"], previsao),
        })
    return pd.DataFrame(resultados)


def main():
    raiz = Path(__file__).resolve().parents[2]
    entrada = raiz / "data/raw/tb_lobby_stats_player.csv"
    saida = raiz / "results/metrics/baseline_kills.csv"
    dados = pd.read_csv(entrada)
    dataset_final = preparar_dataset_final(dados)
    metricas = avaliar_baselines(prever_baselines(dados), dataset_final)
    saida.parent.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(saida, index=False, float_format="%.6f")
    print(metricas.to_string(index=False, float_format=lambda valor: f"{valor:.3f}"))
    print(f"Resultados salvos em {saida}")


if __name__ == "__main__":
    main()
