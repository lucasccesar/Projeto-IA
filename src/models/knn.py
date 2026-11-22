"""KNN (K-Nearest Neighbors) para prever kills a partir do histórico do jogador.

A previsão de uma partida é a média das kills das K partidas do treino com o
histórico mais parecido. Como o KNN compara distâncias, as features são
padronizadas antes, para que nenhuma domine as outras só por ter escala maior.
"""

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.dataset_final import carregar_dataset_final, separar_entradas_alvo
from src.evaluation.metricas import calcular_metricas
from src.evaluation.resultados import formatar_tabela, salvar_tabela
from src.preprocessing.separacao_dados import dividir_por_tempo
from src.utils.caminhos import PASTA_METRICAS

MODELO = "knn"

ARQUIVO_BUSCA = "knn_busca.csv"
ARQUIVO_RESUMO = "knn_kills.csv"

# Hiperparâmetros testados: quantidade de vizinhos e forma de medir a distância.
VALORES_K = [3, 5, 7, 9, 11, 15, 20, 30, 40, 50, 100, 200, 300]
DISTANCIAS = ["euclidean", "manhattan", "chebyshev"]


def criar_modelo(k: int, distancia: str) -> Pipeline:
    """KNN com features padronizadas e todos os vizinhos com o mesmo peso.

    O StandardScaler fica dentro do pipeline, então a média e o desvio são calculados só no treino.
    """
    regressor = KNeighborsRegressor(n_neighbors=k, metric=distancia, weights="uniform", n_jobs=-1)
    return make_pipeline(StandardScaler(), regressor)


def buscar_hiperparametros(
    treino: pd.DataFrame,
    validacao: pd.DataFrame,
    valores_k: list[int] | None = None,
    distancias: list[str] | None = None,
) -> pd.DataFrame:
    """Avalia na validação cada combinação de K e distância.

    Para cada distância, os vizinhos são buscados uma única vez, com o maior K.
    Eles vêm ordenados do mais próximo para o mais distante, então a previsão
    com um K menor é a média das kills dos K primeiros. O resultado é o mesmo
    de treinar um KNN para cada K, mas a busca fica muito mais rápida.

    Retorna uma linha por combinação, da menor para a maior MAE na validação.
    O conjunto de teste não é usado: ele fica reservado para a avaliação final do projeto.
    """
    valores_k = VALORES_K if valores_k is None else valores_k
    distancias = DISTANCIAS if distancias is None else distancias
    X_treino, y_treino = separar_entradas_alvo(treino)
    X_val, y_val = separar_entradas_alvo(validacao)
    if max(valores_k) > len(X_treino):
        raise ValueError("O maior K não pode ser maior que a quantidade de partidas do treino.")

    escala = StandardScaler().fit(X_treino)
    X_treino_padronizado = escala.transform(X_treino)
    X_val_padronizado = escala.transform(X_val)
    kills_treino = y_treino.to_numpy()

    resultados = []
    for distancia in distancias:
        busca = NearestNeighbors(n_neighbors=max(valores_k), metric=distancia, n_jobs=-1)
        vizinhos = busca.fit(X_treino_padronizado).kneighbors(
            X_val_padronizado, return_distance=False
        )
        kills_vizinhos = kills_treino[vizinhos]  # uma linha por partida da validação

        for k in valores_k:
            metricas = calcular_metricas(y_val, kills_vizinhos[:, :k].mean(axis=1))
            resultados.append({
                "k": k,
                "distancia": distancia,
                "MAE_validacao": metricas["MAE"],
                "RMSE_validacao": metricas["RMSE"],
                "R2_validacao": metricas["R2"],
            })
    return (
        pd.DataFrame(resultados)
        .sort_values("MAE_validacao", kind="stable")
        .reset_index(drop=True)
    )


def melhor_configuracao(busca: pd.DataFrame) -> dict:
    """Combinação com o menor MAE na validação."""
    linha = busca.iloc[0]
    return {"k": int(linha["k"]), "distancia": str(linha["distancia"])}


def resumo_validacao(busca: pd.DataFrame, partidas: int) -> pd.DataFrame:
    """Uma linha com as métricas de validação da melhor combinação.

    Segue o mesmo formato do resumo dos demais modelos.
    """
    melhor = busca.iloc[0]
    return pd.DataFrame([{
        "conjunto": "validacao",
        "modelo": MODELO,
        "k": int(melhor["k"]),
        "distancia": str(melhor["distancia"]),
        "partidas": partidas,
        "MAE": melhor["MAE_validacao"],
        "RMSE": melhor["RMSE_validacao"],
        "R2": melhor["R2_validacao"],
    }])


def salvar_resultados(busca: pd.DataFrame, resumo: pd.DataFrame) -> None:
    """Grava a busca completa e o resumo da melhor combinação em results/metrics/."""
    salvar_tabela(busca, ARQUIVO_BUSCA)
    salvar_tabela(resumo, ARQUIVO_RESUMO)


def main() -> None:
    # Lê o dataset final, gerado por `python -m src.data.dataset_final`.
    treino, validacao, _ = dividir_por_tempo(carregar_dataset_final())

    busca = buscar_hiperparametros(treino, validacao)
    resumo = resumo_validacao(busca, len(validacao))
    salvar_resultados(busca, resumo)

    print(formatar_tabela(busca.head(10)))
    print()
    print(formatar_tabela(resumo))
    print(f"Resultados salvos em {PASTA_METRICAS}")


if __name__ == "__main__":
    main()
