"""Busca de hiperparâmetros do KNN usando Particle Swarm Optimization (PSO).

O PSO trabalha com posições contínuas, mas cada dimensão é arredondada para
um índice válido: uma dimensão escolhe K e a outra escolhe a métrica.
A aptidão é o MAE na validação; o conjunto de teste não participa da busca.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset_final import carregar_dataset_final, separar_entradas_alvo
from src.evaluation.metricas import calcular_metricas
from src.evaluation.resultados import formatar_tabela, salvar_tabela
from src.models.knn import DISTANCIAS, VALORES_K, criar_modelo
from src.preprocessing.separacao_dados import dividir_por_tempo
from src.utils.caminhos import PASTA_FIGURAS, PASTA_METRICAS

MODELO = "knn_pso"
TAMANHO_ENXAME = 8
NUM_ITERACOES = 20
INERCIA = 0.7
COEF_COGNITIVO = 1.5
COEF_SOCIAL = 1.5
SEMENTE = 42

ARQUIVO_BUSCA = "knn_pso_busca.csv"
ARQUIVO_RESUMO = "knn_pso_kills.csv"
ARQUIVO_HISTORICO = "knn_pso_convergencia.csv"
ARQUIVO_GRAFICO = "knn_pso_convergencia.png"


def _decodificar_posicao(posicao: np.ndarray) -> tuple[int, str]:
    """Converte coordenadas contínuas nos índices válidos de K e distância."""
    indice_k = int(np.clip(np.rint(posicao[0]), 0, len(VALORES_K) - 1))
    indice_distancia = int(np.clip(np.rint(posicao[1]), 0, len(DISTANCIAS) - 1))
    return VALORES_K[indice_k], DISTANCIAS[indice_distancia]


def buscar_hiperparametros_pso(
    treino: pd.DataFrame,
    validacao: pd.DataFrame,
    tamanho_enxame: int = TAMANHO_ENXAME,
    num_iteracoes: int = NUM_ITERACOES,
    semente: int = SEMENTE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa PSO e retorna configurações avaliadas e histórico de convergência.

    Configurações repetidas são armazenadas em cache, evitando treinar o mesmo
    KNN mais de uma vez. O histórico registra o melhor MAE encontrado até cada
    iteração, inclusive a população inicial (iteração zero).
    """
    if tamanho_enxame < 1 or num_iteracoes < 1:
        raise ValueError("O tamanho do enxame e o número de iterações devem ser positivos.")
    if max(VALORES_K) > len(treino):
        raise ValueError("O maior K não pode ser maior que a quantidade de partidas do treino.")

    X_treino, y_treino = separar_entradas_alvo(treino)
    X_validacao, y_validacao = separar_entradas_alvo(validacao)
    rng = np.random.default_rng(semente)

    limites_inferiores = np.array([0.0, 0.0])
    limites_superiores = np.array([len(VALORES_K) - 1, len(DISTANCIAS) - 1])
    posicoes = rng.uniform(limites_inferiores, limites_superiores, size=(tamanho_enxame, 2))
    velocidades = np.zeros_like(posicoes)

    cache: dict[tuple[int, str], dict] = {}
    historico = []
    melhor_global_posicao = None
    melhor_global_mae = float("inf")

    def avaliar(posicao: np.ndarray) -> dict:
        k, distancia = _decodificar_posicao(posicao)
        chave = (k, distancia)
        if chave not in cache:
            modelo = criar_modelo(k, distancia)
            modelo.fit(X_treino, y_treino)
            metricas = calcular_metricas(y_validacao, modelo.predict(X_validacao))
            cache[chave] = {
                "k": k,
                "distancia": distancia,
                "MAE_validacao": metricas["MAE"],
                "RMSE_validacao": metricas["RMSE"],
                "R2_validacao": metricas["R2"],
            }
        return cache[chave]

    melhores_posicoes = posicoes.copy()
    melhores_maes = np.full(tamanho_enxame, float("inf"))

    for iteracao in range(num_iteracoes + 1):
        for indice, posicao in enumerate(posicoes):
            resultado = avaliar(posicao)
            mae = resultado["MAE_validacao"]

            if mae < melhores_maes[indice]:
                melhores_maes[indice] = mae
                melhores_posicoes[indice] = posicao.copy()

            if mae < melhor_global_mae:
                melhor_global_mae = mae
                melhor_global_posicao = posicao.copy()

        historico.append({
            "iteracao": iteracao,
            "melhor_MAE_validacao": melhor_global_mae,
            "k": _decodificar_posicao(melhor_global_posicao)[0],
            "distancia": _decodificar_posicao(melhor_global_posicao)[1],
            "configuracoes_avaliadas": len(cache),
        })

        if iteracao == num_iteracoes:
            break

        r1 = rng.random(posicoes.shape)
        r2 = rng.random(posicoes.shape)
        velocidades = (
            INERCIA * velocidades
            + COEF_COGNITIVO * r1 * (melhores_posicoes - posicoes)
            + COEF_SOCIAL * r2 * (melhor_global_posicao - posicoes)
        )
        posicoes = np.clip(
            posicoes + velocidades, limites_inferiores, limites_superiores
        )

    busca = (
        pd.DataFrame(cache.values())
        .sort_values("MAE_validacao", kind="stable")
        .reset_index(drop=True)
    )
    return busca, pd.DataFrame(historico)


def resumo_validacao(busca: pd.DataFrame, partidas: int) -> pd.DataFrame:
    """Resume a melhor configuração encontrada pelo PSO."""
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


def salvar_grafico_convergencia(historico: pd.DataFrame) -> None:
    """Salva o gráfico do melhor MAE encontrado ao longo das iterações."""
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        historico["iteracao"],
        historico["melhor_MAE_validacao"],
        marker="o",
    )
    ax.set_title("Convergência do PSO para o KNN")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Melhor MAE na validação")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / ARQUIVO_GRAFICO, dpi=150)
    plt.close(fig)


def main() -> None:
    treino, validacao, _ = dividir_por_tempo(carregar_dataset_final())
    busca, historico = buscar_hiperparametros_pso(treino, validacao)
    resumo = resumo_validacao(busca, len(validacao))

    salvar_tabela(busca, ARQUIVO_BUSCA)
    salvar_tabela(resumo, ARQUIVO_RESUMO)
    salvar_tabela(historico, ARQUIVO_HISTORICO)
    salvar_grafico_convergencia(historico)

    print("Configurações encontradas pelo PSO:")
    print(formatar_tabela(busca))
    print()
    print(formatar_tabela(resumo))
    print(f"Configurações únicas avaliadas: {len(busca)} de {len(VALORES_K) * len(DISTANCIAS)}")
    print(f"Resultados salvos em {PASTA_METRICAS}")
    print(f"Gráfico salvo em {PASTA_FIGURAS / ARQUIVO_GRAFICO}")


if __name__ == "__main__":
    main()
