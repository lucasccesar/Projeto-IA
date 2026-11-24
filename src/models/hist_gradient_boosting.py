"""HistGradientBoostingRegressor para prever kills a partir do histórico do jogador.

A busca de hiperparâmetros usa só o treino e a validação. A avaliação no teste é
um comando separado (`--teste`), que lê a configuração já escolhida na validação.
"""

import argparse
from itertools import product

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data.colunas import TARGET_COLUMN
from src.data.dataset_final import carregar_dataset_final, separar_entradas_alvo
from src.evaluation.metricas import calcular_metricas, metricas_treino_validacao
from src.evaluation.resultados import formatar_tabela, salvar_tabela
from src.preprocessing.separacao_dados import dividir_por_tempo
from src.utils.caminhos import PASTA_METRICAS

SEMENTE = 42
MODELO = "hist_gradient_boosting"
BASELINE = "baseline_media"

ARQUIVO_BUSCA = "hist_gradient_boosting_busca.csv"
ARQUIVO_RESUMO = "hist_gradient_boosting_kills.csv"
ARQUIVO_TESTE = "hist_gradient_boosting_teste.csv"

# Coluna do dataset com a média histórica de kills, usada como baseline no teste.
COLUNA_BASELINE = "mediaKills_hist"

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
TIPOS_PARAMETROS = {
    "loss": str,
    "learning_rate": float,
    "max_iter": int,
    "max_leaf_nodes": int,
    "min_samples_leaf": int,
    "l2_regularization": float,
}


def criar_modelo(
    loss: str = "squared_error",
    learning_rate: float = 0.1,
    max_iter: int = 100,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.0,
) -> HistGradientBoostingRegressor:
    """Gradient Boosting com semente fixa e early stopping desligado.

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


def combinacoes(grade: dict | None = None, losses: tuple[str, ...] = LOSSES) -> list[dict]:
    """Lista de configurações: todas as combinações da grade para cada loss."""
    grade = GRADE if grade is None else grade
    nomes = list(grade)
    return [
        {"loss": loss, **dict(zip(nomes, valores))}
        for loss in losses
        for valores in product(*(grade[nome] for nome in nomes))
    ]


def buscar_hiperparametros(
    treino: pd.DataFrame,
    validacao: pd.DataFrame,
    grade: dict | None = None,
    losses: tuple[str, ...] = LOSSES,
) -> pd.DataFrame:
    """Treina um modelo para cada configuração e avalia no treino e na validação.

    Retorna uma linha por configuração, da menor para a maior MAE na validação.
    O conjunto de teste não é usado: ele fica reservado para a avaliação final do projeto.
    """
    X_treino, y_treino = separar_entradas_alvo(treino)
    X_val, y_val = separar_entradas_alvo(validacao)

    resultados = []
    for configuracao in combinacoes(grade, losses):
        modelo = criar_modelo(**configuracao).fit(X_treino, y_treino)
        metricas = metricas_treino_validacao(modelo, X_treino, y_treino, X_val, y_val)
        resultados.append({
            **configuracao,
            **metricas,
            # Quanto maior, mais o modelo decorou o treino (overfitting).
            "diferenca_MAE": metricas["MAE_validacao"] - metricas["MAE_treino"],
        })
    return (
        pd.DataFrame(resultados)
        .sort_values("MAE_validacao", kind="stable")
        .reset_index(drop=True)
    )


def _como_configuracao(linha: pd.Series) -> dict:
    """Lê os hiperparâmetros de uma linha de tabela com os tipos que o modelo espera."""
    return {nome: TIPOS_PARAMETROS[nome](linha[nome]) for nome in PARAMETROS}


def melhor_configuracao(busca: pd.DataFrame, loss: str | None = None) -> dict:
    """Configuração com o menor MAE na validação (opcionalmente só de uma loss)."""
    candidatas = busca if loss is None else busca[busca["loss"] == loss]
    return _como_configuracao(candidatas.nsmallest(1, "MAE_validacao").iloc[0])


def resumo_validacao(busca: pd.DataFrame, partidas: int) -> pd.DataFrame:
    """Uma linha com a melhor configuração de cada loss, da menor para a maior MAE.

    A primeira linha é a configuração escolhida para o modelo. Segue o mesmo
    formato do resumo dos demais modelos, com a loss como hiperparâmetro.
    """
    linhas = []
    for loss in busca["loss"].drop_duplicates():
        melhor = busca[busca["loss"] == loss].nsmallest(1, "MAE_validacao").iloc[0]
        linhas.append({
            "conjunto": "validacao",
            "modelo": MODELO,
            **_como_configuracao(melhor),
            "partidas": partidas,
            "MAE": melhor["MAE_validacao"],
            "RMSE": melhor["RMSE_validacao"],
            "R2": melhor["R2_validacao"],
        })
    return pd.DataFrame(linhas).sort_values("MAE", kind="stable").reset_index(drop=True)


def configuracao_selecionada(resumo: pd.DataFrame) -> dict:
    """Configuração escolhida na validação: a de menor MAE no resumo."""
    if resumo.empty:
        raise ValueError("O resumo de validação está vazio.")
    return _como_configuracao(resumo.nsmallest(1, "MAE").iloc[0])


def avaliar_no_teste(
    treino: pd.DataFrame, teste: pd.DataFrame, configuracao: dict
) -> pd.DataFrame:
    """Avaliação final da configuração já escolhida na validação.

    O modelo é treinado só no treino, como na busca, e a configuração vem do
    resumo de validação: o teste não participa de nenhuma escolha. A baseline
    (média histórica de kills do jogador) é avaliada nas mesmas partidas.
    """
    X_treino, y_treino = separar_entradas_alvo(treino)
    X_teste, y_teste = separar_entradas_alvo(teste)
    modelo = criar_modelo(**configuracao).fit(X_treino, y_treino)

    linhas = [
        {
            "conjunto": "teste",
            "modelo": BASELINE,
            **{nome: None for nome in PARAMETROS},
            "partidas": len(teste),
            **calcular_metricas(teste[TARGET_COLUMN], teste[COLUNA_BASELINE]),
        },
        {
            "conjunto": "teste",
            "modelo": MODELO,
            **configuracao,
            "partidas": len(teste),
            **calcular_metricas(y_teste, modelo.predict(X_teste)),
        },
    ]
    # A linha da baseline não tem hiperparâmetros; o tipo Int64 aceita o valor vazio
    # sem transformar os inteiros em float (300 em vez de 300.000000 no CSV).
    inteiros = {nome: "Int64" for nome, tipo in TIPOS_PARAMETROS.items() if tipo is int}
    return pd.DataFrame(linhas).astype(inteiros)


def main_teste(treino: pd.DataFrame, teste: pd.DataFrame) -> None:
    """Avalia no teste a configuração salva no resumo de validação."""
    configuracao = configuracao_selecionada(pd.read_csv(PASTA_METRICAS / ARQUIVO_RESUMO))
    resultado = avaliar_no_teste(treino, teste, configuracao)
    salvar_tabela(resultado, ARQUIVO_TESTE)
    print(formatar_tabela(resultado))
    print(f"Resultados salvos em {PASTA_METRICAS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teste",
        action="store_true",
        help="avalia no teste a configuração já selecionada na validação (não refaz a busca)",
    )
    args = parser.parse_args()

    # Lê o dataset final, gerado por `python -m src.data.dataset_final`.
    treino, validacao, teste = dividir_por_tempo(carregar_dataset_final())
    if args.teste:
        main_teste(treino, teste)
        return

    busca = buscar_hiperparametros(treino, validacao)
    resumo = resumo_validacao(busca, len(validacao))
    salvar_tabela(busca, ARQUIVO_BUSCA)
    salvar_tabela(resumo, ARQUIVO_RESUMO)

    print(formatar_tabela(busca.head(10)))
    print()
    print(formatar_tabela(resumo))
    print(f"Resultados salvos em {PASTA_METRICAS}")


if __name__ == "__main__":
    main()
