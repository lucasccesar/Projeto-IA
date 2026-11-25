"""PSO clássico para buscar os hiperparâmetros do HistGradientBoostingRegressor.

Cada partícula é um vetor com os 6 hiperparâmetros do modelo. O fitness é o MAE
na validação, que o PSO minimiza. O conjunto de teste não participa da busca.

Com `--teste`, avalia no conjunto de teste a configuração já encontrada pelo PSO
(lida de `hist_gradient_boosting_pso_kills.csv`), sem refazer a busca.

O PSO é a versão *global best* com peso de inércia:

    v = w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)
    x = x + v,  com r1, r2 ~ U(0, 1) sorteados por partícula e por dimensão.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.data.dataset_final import carregar_dataset_final, separar_entradas_alvo
from src.evaluation.metricas import calcular_metricas
from src.evaluation.resultados import CASAS_ARQUIVO, formatar_tabela, salvar_tabela
from src.models.hist_gradient_boosting import (
    ARQUIVO_RESUMO as ARQUIVO_RESUMO_GRADE,
    LOSSES,
    PARAMETROS,
    SEMENTE,
    TIPOS_PARAMETROS,
    avaliar_no_teste,
    configuracao_selecionada,
    criar_modelo,
)
from src.preprocessing.separacao_dados import dividir_por_tempo
from src.utils.caminhos import PASTA_FIGURAS, PASTA_METRICAS

MODELO = "hist_gradient_boosting_pso"

ARQUIVO_HISTORICO = "hist_gradient_boosting_pso_historico.csv"
ARQUIVO_RESUMO = "hist_gradient_boosting_pso_kills.csv"
ARQUIVO_TESTE = "hist_gradient_boosting_pso_teste.csv"
ARQUIVO_FIGURA = "hist_gradient_boosting_pso_convergencia.png"

# Coeficientes da literatura (Eberhart e Shi, 2000; Clerc e Kennedy, 2002).
# Com w < 1 e w > (c1 + c2)/2 − 1, as partículas convergem sem a velocidade explodir;
# c1 = c2 dá o mesmo peso à melhor posição da partícula e à melhor do enxame.
W = 0.7298  # inércia
C1 = 1.49618  # força cognitiva (puxa para o pbest)
C2 = 1.49618  # força social (puxa para o gbest)

N_PARTICULAS = 10
N_ITERACOES = 10
POPULACAO_MINIMA = 10

# Limites de cada dimensão do vetor, na ordem de PARAMETROS. Contêm os valores da
# grid search e se estendem na direção em que o MAE melhorou (o melhor da grade
# ficou na borda em todos os hiperparâmetros numéricos).
# A loss é codificada como número: posição < 0,5 → squared_error; ≥ 0,5 → absolute_error.
LIMITES = {
    "loss": (0.0, 1.0),
    "learning_rate": (0.01, 0.3),
    "max_iter": (50, 500),
    "max_leaf_nodes": (7, 63),
    "min_samples_leaf": (10, 500),
    "l2_regularization": (0.0, 2.0),
}


@dataclass
class ResultadoPSO:
    """Resultado de uma execução do PSO.

    `historico` tem uma entrada por iteração (a 0 é a população inicial), com o
    fitness médio do enxame, o fitness do gbest e a posição do gbest naquele momento.
    """

    posicao: np.ndarray
    fitness: float
    historico: list[dict]


def _matriz_limites(limites: dict[str, tuple[float, float]]) -> np.ndarray:
    """Limites como matriz (dimensões × 2), na ordem de PARAMETROS."""
    if tuple(limites) != PARAMETROS:
        raise ValueError(f"Os limites devem seguir a ordem {PARAMETROS}.")
    return np.array(list(limites.values()), dtype=float)


def inicializar_posicoes(
    n_particulas: int, limites: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Posições iniciais sorteadas com distribuição uniforme dentro dos limites."""
    minimo, maximo = limites[:, 0], limites[:, 1]
    return minimo + rng.random((n_particulas, len(limites))) * (maximo - minimo)


def otimizar_pso(
    fitness: Callable[[np.ndarray], np.ndarray],
    limites: np.ndarray,
    n_particulas: int = N_PARTICULAS,
    n_iteracoes: int = N_ITERACOES,
    w: float = W,
    c1: float = C1,
    c2: float = C2,
    semente: int = SEMENTE,
) -> ResultadoPSO:
    """Minimiza `fitness` com o PSO clássico (global best, com inércia).

    `fitness` recebe as posições de todo o enxame (partículas × dimensões) e
    retorna um valor por partícula. As velocidades começam em zero. Uma partícula
    que sai dos limites volta para a borda e tem a velocidade daquela dimensão
    zerada, para não ficar presa empurrando contra o limite.
    """
    if n_particulas < POPULACAO_MINIMA:
        raise ValueError(
            f"A população deve ter pelo menos {POPULACAO_MINIMA} partículas "
            f"(recebido: {n_particulas})."
        )
    rng = np.random.default_rng(semente)
    minimo, maximo = limites[:, 0], limites[:, 1]
    velocidade_maxima = maximo - minimo

    posicoes = inicializar_posicoes(n_particulas, limites, rng)
    velocidades = np.zeros_like(posicoes)
    valores = np.asarray(fitness(posicoes), dtype=float)

    pbest, pbest_valores = posicoes.copy(), valores.copy()
    indice = int(np.argmin(pbest_valores))
    gbest, gbest_valor = pbest[indice].copy(), float(pbest_valores[indice])
    historico = [_registro(0, valores, gbest, gbest_valor)]

    for iteracao in range(1, n_iteracoes + 1):
        r1 = rng.random(posicoes.shape)
        r2 = rng.random(posicoes.shape)
        velocidades = (
            w * velocidades
            + c1 * r1 * (pbest - posicoes)
            + c2 * r2 * (gbest - posicoes)
        )
        velocidades = np.clip(velocidades, -velocidade_maxima, velocidade_maxima)
        posicoes = posicoes + velocidades

        fora = (posicoes < minimo) | (posicoes > maximo)
        posicoes = np.clip(posicoes, minimo, maximo)
        velocidades[fora] = 0.0

        valores = np.asarray(fitness(posicoes), dtype=float)
        melhorou = valores < pbest_valores
        pbest[melhorou] = posicoes[melhorou]
        pbest_valores[melhorou] = valores[melhorou]

        indice = int(np.argmin(pbest_valores))
        if pbest_valores[indice] < gbest_valor:
            gbest, gbest_valor = pbest[indice].copy(), float(pbest_valores[indice])
        historico.append(_registro(iteracao, valores, gbest, gbest_valor))

    return ResultadoPSO(posicao=gbest, fitness=gbest_valor, historico=historico)


def _registro(
    iteracao: int, valores: np.ndarray, gbest: np.ndarray, gbest_valor: float
) -> dict:
    """Estado do enxame ao fim de uma iteração."""
    return {
        "iteracao": iteracao,
        "fitness_medio": float(np.mean(valores)),
        "fitness_gbest": gbest_valor,
        "gbest": gbest.copy(),
    }


def vetor_para_configuracao(
    posicao: np.ndarray, limites: dict[str, tuple[float, float]] = LIMITES
) -> dict:
    """Converte a posição de uma partícula nos hiperparâmetros do modelo.

    A posição é limitada aos limites de cada dimensão; a loss é escolhida por
    arredondamento (0 → squared_error, 1 → absolute_error), os hiperparâmetros
    inteiros são arredondados para o inteiro mais próximo e os contínuos para as
    casas decimais gravadas nos CSVs. Assim, a configuração lida do CSV é
    exatamente a que foi treinada e avaliada.
    """
    matriz = _matriz_limites(limites)
    valores = np.clip(np.asarray(posicao, dtype=float), matriz[:, 0], matriz[:, 1])

    configuracao = {}
    for nome, valor in zip(PARAMETROS, valores):
        if nome == "loss":
            configuracao[nome] = LOSSES[int(valor >= 0.5)]
        elif TIPOS_PARAMETROS[nome] is int:
            configuracao[nome] = int(round(valor))
        else:
            configuracao[nome] = round(float(valor), CASAS_ARQUIVO)
    return configuracao


class AvaliadorHGB:
    """Fitness do PSO: MAE na validação do modelo treinado só no treino.

    Guarda as métricas de cada configuração já avaliada. Partículas diferentes que
    viram a mesma configuração (depois do arredondamento) não treinam de novo.
    """

    def __init__(
        self,
        treino: pd.DataFrame,
        validacao: pd.DataFrame,
        limites: dict[str, tuple[float, float]] = LIMITES,
        mostrar_progresso: bool = False,
    ) -> None:
        self.X_treino, self.y_treino = separar_entradas_alvo(treino)
        self.X_val, self.y_val = separar_entradas_alvo(validacao)
        self.limites = limites
        self.mostrar_progresso = mostrar_progresso
        self.metricas: dict[tuple, dict[str, float]] = {}
        # Total de treinos feitos até o fim de cada chamada (uma chamada por iteração).
        self.treinos_acumulados: list[int] = []

    def metricas_da_configuracao(self, configuracao: dict) -> dict[str, float]:
        """MAE, RMSE e R² na validação, treinando o modelo só se ainda não foi treinado."""
        chave = tuple(configuracao[nome] for nome in PARAMETROS)
        if chave not in self.metricas:
            modelo = criar_modelo(**configuracao).fit(self.X_treino, self.y_treino)
            self.metricas[chave] = calcular_metricas(self.y_val, modelo.predict(self.X_val))
        return self.metricas[chave]

    def __call__(self, posicoes: np.ndarray) -> np.ndarray:
        """MAE na validação de cada partícula do enxame."""
        maes = np.array([
            self.metricas_da_configuracao(vetor_para_configuracao(p, self.limites))["MAE"]
            for p in posicoes
        ])
        self.treinos_acumulados.append(len(self.metricas))
        if self.mostrar_progresso:
            print(
                f"Iteração {len(self.treinos_acumulados) - 1}: menor MAE {maes.min():.4f}, "
                f"{len(self.metricas)} treinos até agora"
            )
        return maes


def buscar_hiperparametros(
    treino: pd.DataFrame,
    validacao: pd.DataFrame,
    limites: dict[str, tuple[float, float]] = LIMITES,
    n_particulas: int = N_PARTICULAS,
    n_iteracoes: int = N_ITERACOES,
    mostrar_progresso: bool = False,
) -> tuple[dict, dict[str, float], pd.DataFrame]:
    """Roda o PSO e retorna a melhor configuração, suas métricas e o histórico.

    O histórico tem uma linha por iteração (a 0 é a população inicial), com o total
    de treinos, o MAE médio do enxame, o MAE do gbest e a configuração do gbest.
    """
    avaliador = AvaliadorHGB(treino, validacao, limites, mostrar_progresso)
    resultado = otimizar_pso(
        avaliador, _matriz_limites(limites), n_particulas=n_particulas, n_iteracoes=n_iteracoes
    )

    historico = pd.DataFrame([
        {
            "iteracao": registro["iteracao"],
            "treinos": treinos,
            "MAE_medio": registro["fitness_medio"],
            "MAE_gbest": registro["fitness_gbest"],
            **vetor_para_configuracao(registro["gbest"], limites),
        }
        for registro, treinos in zip(resultado.historico, avaliador.treinos_acumulados)
    ])
    configuracao = vetor_para_configuracao(resultado.posicao, limites)
    return configuracao, avaliador.metricas_da_configuracao(configuracao), historico


def resumo_validacao(
    configuracao: dict, metricas: dict[str, float], partidas: int
) -> pd.DataFrame:
    """Uma linha com a configuração encontrada pelo PSO, no formato dos demais modelos."""
    return pd.DataFrame([{
        "conjunto": "validacao",
        "modelo": MODELO,
        **configuracao,
        "partidas": partidas,
        **metricas,
    }])


def grafico_convergencia(
    historico: pd.DataFrame, caminho: str | Path, mae_referencia: float | None = None
) -> None:
    """Grava o gráfico de convergência do PSO.

    À esquerda, o MAE do gbest por iteração, comparado com o melhor MAE da grid
    search. À direita, o MAE médio do enxame junto com o do gbest: quando as duas
    curvas se aproximam, as partículas se concentraram perto da melhor posição.
    """
    fig = Figure(figsize=(12, 4.5))
    eixo_gbest, eixo_enxame = fig.subplots(1, 2)
    iteracoes = historico["iteracao"]

    eixo_gbest.plot(iteracoes, historico["MAE_gbest"], marker="o", color="#2563eb",
                    label="gbest (PSO)")
    if mae_referencia is not None:
        eixo_gbest.axhline(mae_referencia, color="#94a3b8", linestyle="--",
                           label=f"Grid search ({mae_referencia:.3f})")
    eixo_gbest.set_title("MAE do gbest na validação")

    eixo_enxame.plot(iteracoes, historico["MAE_medio"], marker="o", color="#d97706",
                     label="Média do enxame")
    eixo_enxame.plot(iteracoes, historico["MAE_gbest"], marker="o", color="#2563eb",
                     label="gbest")
    eixo_enxame.set_title("MAE médio do enxame na validação")

    for eixo in (eixo_gbest, eixo_enxame):
        eixo.set_xlabel("Iteração (0 = população inicial)")
        eixo.set_ylabel("MAE (kills)")
        eixo.set_xticks(iteracoes)
        eixo.grid(alpha=0.3)
        eixo.legend()
    fig.suptitle("HistGradientBoosting: convergência do PSO")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)


def mae_da_grid_search(pasta: Path = PASTA_METRICAS) -> float | None:
    """Melhor MAE da grid search do HGB, usado como referência (None se não houver)."""
    caminho = pasta / ARQUIVO_RESUMO_GRADE
    return float(pd.read_csv(caminho)["MAE"].min()) if caminho.is_file() else None


def main_teste(treino: pd.DataFrame, teste: pd.DataFrame) -> None:
    """Avalia no teste a configuração salva no resumo de validação do PSO."""
    configuracao = configuracao_selecionada(pd.read_csv(PASTA_METRICAS / ARQUIVO_RESUMO))
    resultado = avaliar_no_teste(treino, teste, configuracao, nome_modelo=MODELO)
    salvar_tabela(resultado, ARQUIVO_TESTE)
    print(formatar_tabela(resultado))
    print(f"Resultados salvos em {PASTA_METRICAS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teste",
        action="store_true",
        help="avalia no teste a configuração já encontrada pelo PSO (não refaz a busca)",
    )
    args = parser.parse_args()

    # Lê o dataset final, gerado por `python -m src.data.dataset_final`.
    treino, validacao, teste = dividir_por_tempo(carregar_dataset_final())
    if args.teste:
        main_teste(treino, teste)
        return

    configuracao, metricas, historico = buscar_hiperparametros(
        treino, validacao, mostrar_progresso=True
    )
    resumo = resumo_validacao(configuracao, metricas, len(validacao))

    salvar_tabela(historico, ARQUIVO_HISTORICO)
    salvar_tabela(resumo, ARQUIVO_RESUMO)
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    grafico_convergencia(historico, PASTA_FIGURAS / ARQUIVO_FIGURA, mae_da_grid_search())

    print()
    print(formatar_tabela(historico))
    print()
    print(formatar_tabela(resumo))
    print(f"Resultados salvos em {PASTA_METRICAS} e {PASTA_FIGURAS}")


if __name__ == "__main__":
    main()
