"""Árvore de Decisão para prever kills a partir do histórico do jogador."""

from itertools import product
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from sklearn.tree import DecisionTreeRegressor

from src.data.dataset_final import FEATURE_COLUMNS, carregar_dataset_final, separar_entradas_alvo
from src.evaluation.metricas import calcular_metricas, metricas_treino_validacao
from src.evaluation.resultados import formatar_tabela, salvar_tabela
from src.preprocessing.separacao_dados import dividir_por_tempo
from src.utils.caminhos import PASTA_FIGURAS, PASTA_METRICAS

SEMENTE = 42
MODELO = "arvore_decisao"

ARQUIVO_BUSCA = "arvore_decisao_busca.csv"
ARQUIVO_RESUMO = "arvore_decisao_kills.csv"
ARQUIVO_IMPORTANCIAS = "arvore_decisao_importancias.csv"
ARQUIVO_FIGURA = "arvore_decisao_overfitting.png"

# Texto usado nas tabelas para max_depth=None (árvore sem limite de profundidade).
SEM_LIMITE = "None"

# Hiperparâmetros testados. Os dois controlam o tamanho da árvore,
# que é o principal fator de overfitting em uma árvore isolada.
GRADE = {
    "max_depth": [2, 4, 6, 8, 10, 15, 20, None],
    "min_samples_leaf": [1, 10, 50, 200, 1000],
}


def criar_modelo(max_depth: int | None = None, min_samples_leaf: int = 1) -> DecisionTreeRegressor:
    """Árvore de regressão com semente fixa, para que os resultados sejam reproduzíveis."""
    return DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=SEMENTE
    )


def _rotulo_profundidade(max_depth: int | None) -> int | str:
    """Valor de max_depth como aparece nas tabelas (SEM_LIMITE quando é None)."""
    return SEM_LIMITE if max_depth is None else max_depth


def buscar_hiperparametros(
    treino: pd.DataFrame, validacao: pd.DataFrame, grade: dict | None = None
) -> pd.DataFrame:
    """Treina uma árvore para cada combinação da grade e avalia no treino e na validação.

    Retorna uma linha por combinação, da menor para a maior MAE na validação.
    O conjunto de teste não é usado: ele fica reservado para a avaliação final do projeto.
    """
    grade = GRADE if grade is None else grade
    X_treino, y_treino = separar_entradas_alvo(treino)
    X_val, y_val = separar_entradas_alvo(validacao)

    resultados = []
    for max_depth, min_samples_leaf in product(grade["max_depth"], grade["min_samples_leaf"]):
        modelo = criar_modelo(max_depth, min_samples_leaf).fit(X_treino, y_treino)
        metricas = metricas_treino_validacao(modelo, X_treino, y_treino, X_val, y_val)
        resultados.append({
            "max_depth": _rotulo_profundidade(max_depth),
            "min_samples_leaf": min_samples_leaf,
            "profundidade_real": modelo.get_depth(),
            "folhas": modelo.get_n_leaves(),
            **metricas,
            # Quanto maior, mais a árvore decorou o treino (overfitting).
            "diferenca_MAE": metricas["MAE_validacao"] - metricas["MAE_treino"],
        })
    return pd.DataFrame(resultados).sort_values("MAE_validacao").reset_index(drop=True)


def melhor_configuracao(busca: pd.DataFrame) -> dict:
    """Combinação com o menor MAE na validação."""
    linha = busca.iloc[0]
    max_depth = None if linha["max_depth"] == SEM_LIMITE else int(linha["max_depth"])
    return {"max_depth": max_depth, "min_samples_leaf": int(linha["min_samples_leaf"])}


def avaliar_melhor_modelo(
    treino: pd.DataFrame, validacao: pd.DataFrame, configuracao: dict
) -> tuple[DecisionTreeRegressor, pd.DataFrame, pd.DataFrame]:
    """Treina a melhor configuração e avalia na validação.

    Retorna o modelo treinado, as métricas na validação e a importância das features.
    """
    X_treino, y_treino = separar_entradas_alvo(treino)
    X_val, y_val = separar_entradas_alvo(validacao)
    modelo = criar_modelo(**configuracao).fit(X_treino, y_treino)

    resumo = pd.DataFrame([{
        "conjunto": "validacao",
        "modelo": MODELO,
        "max_depth": _rotulo_profundidade(configuracao["max_depth"]),
        "min_samples_leaf": configuracao["min_samples_leaf"],
        "partidas": len(validacao),
        **calcular_metricas(y_val, modelo.predict(X_val)),
    }])
    importancias = (
        pd.DataFrame({"feature": list(FEATURE_COLUMNS), "importancia": modelo.feature_importances_})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    return modelo, resumo, importancias


def grafico_overfitting(
    busca: pd.DataFrame, caminho: str | Path, folhas: tuple[int, int] = (1, 200)
) -> None:
    """Grava o gráfico do MAE no treino e na validação conforme a profundidade.

    Mostra uma curva contínua e outra tracejada, uma para cada valor de
    min_samples_leaf em `folhas`. As profundidades vêm da própria tabela de busca.
    """
    dados = busca.assign(profundidade=busca["max_depth"].astype(str))
    profundidades = set(dados["profundidade"])
    ordem = sorted(profundidades - {SEM_LIMITE}, key=int)
    if SEM_LIMITE in profundidades:
        ordem.append(SEM_LIMITE)  # a árvore sem limite fica no fim do eixo

    fig = Figure(figsize=(8, 4.5))
    ax = fig.subplots()
    for folha, estilo in zip(folhas, ("-", "--")):
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


def main() -> None:
    # Lê o dataset final, gerado por `python -m src.data.dataset_final`.
    treino, validacao, _ = dividir_por_tempo(carregar_dataset_final())

    busca = buscar_hiperparametros(treino, validacao)
    configuracao = melhor_configuracao(busca)
    _, resumo, importancias = avaliar_melhor_modelo(treino, validacao, configuracao)

    salvar_tabela(busca, ARQUIVO_BUSCA)
    salvar_tabela(resumo, ARQUIVO_RESUMO)
    salvar_tabela(importancias, ARQUIVO_IMPORTANCIAS)
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    grafico_overfitting(busca, PASTA_FIGURAS / ARQUIVO_FIGURA)

    print(formatar_tabela(busca.head(10)))
    print()
    print(formatar_tabela(resumo))
    print(f"Resultados salvos em {PASTA_METRICAS} e {PASTA_FIGURAS}")


if __name__ == "__main__":
    main()
