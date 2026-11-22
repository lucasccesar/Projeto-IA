"""HistGradientBoostingRegressor para prever kills a partir do histórico do jogador."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.dataset_final import FEATURE_COLUMNS, TARGET_COLUMN, carregar_dataset_final
from src.preprocessing.separacao_dados import dividir_por_tempo


SEMENTE = 42
LOSSES = ("squared_error", "absolute_error")

# Hiperparâmetros testados para cada loss (32 combinações por loss).
# - learning_rate e max_iter controlam quanto o modelo ajusta no total;
# - max_leaf_nodes e min_samples_leaf controlam o tamanho de cada árvore
#   (na árvore isolada, folhas grandes reduziram bastante o overfitting);
# - l2_regularization encolhe os valores das folhas.
# max_depth fica sem limite porque o tamanho já é limitado por max_leaf_nodes.
GRADE = {
    "learning_rate": [0.05, 0.1],
    "max_iter": [100, 300],
    "max_leaf_nodes": [15, 31],
    "min_samples_leaf": [20, 200],
    "l2_regularization": [0.0, 1.0],
}
PARAMETROS = ("loss", *GRADE)


def separar_entradas_alvo(df):
    """Retorna as features de entrada (X) e as kills (y)."""
    return df[list(FEATURE_COLUMNS)], df[TARGET_COLUMN]


def criar_modelo(loss="squared_error", learning_rate=0.1, max_iter=100, max_leaf_nodes=31,
                 min_samples_leaf=20, l2_regularization=0.0):
    """Gradient Boosting com early stopping desligado.

    O early stopping do scikit-learn separa uma amostra aleatória do treino, o que
    misturaria partidas de datas diferentes. Por isso o número de iterações é fixo
    e entra na grade de hiperparâmetros.
    """
    return HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=False,
        random_state=SEMENTE,
    )


def calcular_metricas(y_real, y_previsto):
    """MAE, RMSE e R² em kills."""
    return {
        "MAE": mean_absolute_error(y_real, y_previsto),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_previsto))),
        "R2": r2_score(y_real, y_previsto),
    }


def combinacoes(grade=None, losses=LOSSES):
    """Lista de configurações: todas as combinações da grade para cada loss."""
    grade = GRADE if grade is None else grade
    nomes = list(grade)
    return [
        {"loss": loss, **dict(zip(nomes, valores))}
        for loss in losses
        for valores in product(*(grade[nome] for nome in nomes))
    ]


def buscar_hiperparametros(treino, validacao, grade=None, losses=LOSSES):
    """Treina um modelo para cada configuração e avalia no treino e na validação.

    O conjunto de teste não é usado: ele fica reservado para a avaliação final do projeto.
    """
    X_treino, y_treino = separar_entradas_alvo(treino)
    X_val, y_val = separar_entradas_alvo(validacao)

    resultados = []
    for configuracao in combinacoes(grade, losses):
        modelo = criar_modelo(**configuracao).fit(X_treino, y_treino)
        m_treino = calcular_metricas(y_treino, modelo.predict(X_treino))
        m_val = calcular_metricas(y_val, modelo.predict(X_val))
        resultados.append({
            **configuracao,
            "partidas_treino": len(treino),
            "partidas_validacao": len(validacao),
            "MAE_treino": m_treino["MAE"],
            "MAE_validacao": m_val["MAE"],
            "RMSE_validacao": m_val["RMSE"],
            "R2_validacao": m_val["R2"],
            "diferenca_MAE": m_val["MAE"] - m_treino["MAE"],
        })
    return pd.DataFrame(resultados).sort_values("MAE_validacao", kind="stable").reset_index(drop=True)


def _como_configuracao(linha):
    tipos = {"loss": str, "learning_rate": float, "max_iter": int, "max_leaf_nodes": int,
             "min_samples_leaf": int, "l2_regularization": float}
    return {nome: tipos[nome](linha[nome]) for nome in PARAMETROS}


def melhor_configuracao(busca, loss=None):
    """Configuração com o menor MAE na validação (opcionalmente só de uma loss)."""
    candidatas = busca if loss is None else busca[busca["loss"] == loss]
    return _como_configuracao(candidatas.nsmallest(1, "MAE_validacao").iloc[0])


def resumo_validacao(busca):
    """Uma linha com a melhor configuração de cada loss; `selecionado` marca a de menor MAE."""
    escolhida = melhor_configuracao(busca)
    linhas = []
    for loss in busca["loss"].drop_duplicates():
        linha = busca[busca["loss"] == loss].nsmallest(1, "MAE_validacao").iloc[0]
        configuracao = _como_configuracao(linha)
        linhas.append({
            "conjunto": "validacao",
            "modelo": f"hist_gradient_boosting_{loss}",
            **configuracao,
            "partidas": int(linha["partidas_validacao"]),
            "partidas_treino": int(linha["partidas_treino"]),
            "MAE": linha["MAE_validacao"],
            "RMSE": linha["RMSE_validacao"],
            "R2": linha["R2_validacao"],
            "selecionado": configuracao == escolhida,
        })
    return pd.DataFrame(linhas).sort_values("MAE", kind="stable").reset_index(drop=True)


def main():
    raiz = Path(__file__).resolve().parents[2]
    pasta = raiz / "results/metrics"
    pasta.mkdir(parents=True, exist_ok=True)

    # Lê data/processed/tb_lobby_stats_player_final.csv (gerado por python -m src.data.dataset_final)
    dataset_final = carregar_dataset_final()
    treino, validacao, _ = dividir_por_tempo(dataset_final)

    busca = buscar_hiperparametros(treino, validacao)
    resumo = resumo_validacao(busca)

    busca.to_csv(pasta / "hist_gradient_boosting_busca.csv", index=False, float_format="%.6f")
    resumo.to_csv(pasta / "hist_gradient_boosting_kills.csv", index=False, float_format="%.6f")

    formato = lambda v: f"{v:.3f}"
    print(busca.head(10).to_string(index=False, float_format=formato))
    print()
    print(resumo.to_string(index=False, float_format=formato))
    print(f"Resultados salvos em {pasta}")


if __name__ == "__main__":
    main()
