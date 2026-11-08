"""Árvore de Decisão para prever kills a partir do histórico do jogador."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from src.data.dataset_final import FEATURE_COLUMNS, TARGET_COLUMN, carregar_dataset_final
from src.preprocessing.separacao_dados import dividir_por_tempo


SEMENTE = 42

# Hiperparâmetros testados. Os dois controlam o tamanho da árvore,
# que é o principal fator de overfitting em uma árvore isolada.
GRADE = {
    "max_depth": [2, 4, 6, 8, 10, 15, 20, None],
    "min_samples_leaf": [1, 10, 50, 200, 1000],
}


def separar_entradas_alvo(df):
    """Retorna as features de entrada (X) e as kills (y)."""
    return df[list(FEATURE_COLUMNS)], df[TARGET_COLUMN]


def criar_modelo(max_depth=None, min_samples_leaf=1):
    return DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=SEMENTE
    )


def calcular_metricas(y_real, y_previsto):
    """MAE, RMSE e R² em kills."""
    return {
        "MAE": mean_absolute_error(y_real, y_previsto),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_previsto))),
        "R2": r2_score(y_real, y_previsto),
    }


def buscar_hiperparametros(treino, validacao, grade=None):
    """Treina uma árvore para cada combinação da grade e avalia no treino e na validação.

    O conjunto de teste não é usado: ele fica reservado para a avaliação final do projeto.
    """
    grade = GRADE if grade is None else grade
    X_treino, y_treino = separar_entradas_alvo(treino)
    X_val, y_val = separar_entradas_alvo(validacao)

    resultados = []
    for max_depth, min_samples_leaf in product(grade["max_depth"], grade["min_samples_leaf"]):
        modelo = criar_modelo(max_depth, min_samples_leaf).fit(X_treino, y_treino)
        m_treino = calcular_metricas(y_treino, modelo.predict(X_treino))
        m_val = calcular_metricas(y_val, modelo.predict(X_val))
        resultados.append({
            "max_depth": "None" if max_depth is None else max_depth,
            "min_samples_leaf": min_samples_leaf,
            "profundidade_real": modelo.get_depth(),
            "folhas": modelo.get_n_leaves(),
            "MAE_treino": m_treino["MAE"],
            "MAE_validacao": m_val["MAE"],
            "RMSE_validacao": m_val["RMSE"],
            "R2_validacao": m_val["R2"],
            "diferenca_MAE": m_val["MAE"] - m_treino["MAE"],
        })
    return pd.DataFrame(resultados).sort_values("MAE_validacao").reset_index(drop=True)


def melhor_configuracao(busca):
    """Combinação com o menor MAE na validação."""
    linha = busca.iloc[0]
    max_depth = None if linha["max_depth"] == "None" else int(linha["max_depth"])
    return {"max_depth": max_depth, "min_samples_leaf": int(linha["min_samples_leaf"])}


def avaliar_melhor_modelo(treino, validacao, configuracao):
    """Treina a melhor configuração e retorna métricas na validação e a importância das features."""
    X_treino, y_treino = separar_entradas_alvo(treino)
    X_val, y_val = separar_entradas_alvo(validacao)
    modelo = criar_modelo(**configuracao).fit(X_treino, y_treino)

    metricas = calcular_metricas(y_val, modelo.predict(X_val))
    resumo = pd.DataFrame([{
        "conjunto": "validacao",
        "modelo": "arvore_decisao",
        "max_depth": "None" if configuracao["max_depth"] is None else configuracao["max_depth"],
        "min_samples_leaf": configuracao["min_samples_leaf"],
        "partidas": len(validacao),
        **metricas,
    }])
    importancias = (
        pd.DataFrame({"feature": list(FEATURE_COLUMNS), "importancia": modelo.feature_importances_})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    return modelo, resumo, importancias


def grafico_overfitting(busca, caminho):
    """MAE no treino e na validação conforme a profundidade, para min_samples_leaf 1 e 200."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dados = busca.copy()
    dados["profundidade"] = dados["max_depth"].astype(str)
    ordem = [str(d) for d in GRADE["max_depth"] if d is not None] + ["None"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for folha, estilo in [(1, "-"), (200, "--")]:
        linha = dados[dados["min_samples_leaf"] == folha].set_index("profundidade").loc[ordem]
        ax.plot(ordem, linha["MAE_treino"], estilo, marker="o", color="#1f77b4",
                label=f"Treino (min_samples_leaf={folha})")
        ax.plot(ordem, linha["MAE_validacao"], estilo, marker="o", color="#d62728",
                label=f"Validação (min_samples_leaf={folha})")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("MAE (kills)")
    ax.set_title("Árvore de Decisão: erro no treino x validação")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def main():
    raiz = Path(__file__).resolve().parents[2]
    pasta = raiz / "results/metrics"
    figuras = raiz / "results/figures"
    pasta.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)

    # Lê data/processed/tb_lobby_stats_player_final.csv (gerado por python -m src.data.dataset_final)
    dataset_final = carregar_dataset_final()
    treino, validacao, _ = dividir_por_tempo(dataset_final)

    busca = buscar_hiperparametros(treino, validacao)
    configuracao = melhor_configuracao(busca)
    _, resumo, importancias = avaliar_melhor_modelo(treino, validacao, configuracao)

    busca.to_csv(pasta / "arvore_decisao_busca.csv", index=False, float_format="%.6f")
    resumo.to_csv(pasta / "arvore_decisao_kills.csv", index=False, float_format="%.6f")
    importancias.to_csv(pasta / "arvore_decisao_importancias.csv", index=False, float_format="%.6f")
    grafico_overfitting(busca, figuras / "arvore_decisao_overfitting.png")

    print(busca.head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(resumo.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Resultados salvos em {pasta} e {figuras}")


if __name__ == "__main__":
    main()
