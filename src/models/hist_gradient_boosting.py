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


def configuracao_selecionada(resumo):
    """Lê a configuração marcada como selecionada no resumo de validação."""
    selecionadas = resumo[resumo["selecionado"].astype(bool)]
    if len(selecionadas) != 1:
        raise ValueError("O resumo de validação deve ter exatamente uma configuração selecionada.")
    linha = selecionadas.iloc[0]
    return _como_configuracao(linha), float(linha["MAE"])


def avaliar_no_teste(treino, validacao, teste, configuracao, mae_validacao):
    """Avaliação final da configuração já escolhida na validação.

    O modelo é treinado só no treino, como na busca, e a configuração vem do
    resumo de validação: o teste não participa de nenhuma escolha. A baseline
    (média histórica de kills do jogador) é avaliada nas mesmas partidas.
    """
    X_treino, y_treino = separar_entradas_alvo(treino)
    X_teste, y_teste = separar_entradas_alvo(teste)
    modelo = criar_modelo(**configuracao).fit(X_treino, y_treino)
    m_modelo = calcular_metricas(y_teste, modelo.predict(X_teste))
    m_baseline = calcular_metricas(y_teste, teste["mediaKills_hist"])
    mae_baseline_val = mean_absolute_error(validacao[TARGET_COLUMN], validacao["mediaKills_hist"])

    linhas = [
        {"modelo": "baseline (média histórica)", **{nome: "" for nome in PARAMETROS},
         "MAE_validacao": mae_baseline_val, **m_baseline},
        {"modelo": f"hist_gradient_boosting_{configuracao['loss']}", **configuracao,
         "MAE_validacao": mae_validacao, **m_modelo},
    ]
    resultado = pd.DataFrame(linhas).rename(
        columns={"MAE": "MAE_teste", "RMSE": "RMSE_teste", "R2": "R2_teste"}
    )
    resultado.insert(1, "conjunto", "teste")
    resultado.insert(len(PARAMETROS) + 2, "partidas_treino", len(treino))
    resultado.insert(len(PARAMETROS) + 3, "partidas_teste", len(teste))
    resultado["diferenca_para_baseline"] = resultado["MAE_teste"] - m_baseline["MAE"]
    return resultado


def main_teste(pasta, treino, validacao, teste):
    """Avalia no teste a configuração salva em hist_gradient_boosting_kills.csv."""
    configuracao, mae_validacao = configuracao_selecionada(
        pd.read_csv(pasta / "hist_gradient_boosting_kills.csv")
    )
    resultado = avaliar_no_teste(treino, validacao, teste, configuracao, mae_validacao)
    resultado.to_csv(pasta / "hist_gradient_boosting_teste.csv", index=False, float_format="%.6f")
    print(resultado.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Resultados salvos em {pasta}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teste", action="store_true",
        help="avalia no teste a configuração já selecionada na validação (não refaz a busca)",
    )
    args = parser.parse_args()

    raiz = Path(__file__).resolve().parents[2]
    pasta = raiz / "results/metrics"
    pasta.mkdir(parents=True, exist_ok=True)

    # Lê data/processed/tb_lobby_stats_player_final.csv (gerado por python -m src.data.dataset_final)
    dataset_final = carregar_dataset_final()
    treino, validacao, teste = dividir_por_tempo(dataset_final)
    if args.teste:
        main_teste(pasta, treino, validacao, teste)
        return

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
