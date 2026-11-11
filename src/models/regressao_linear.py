"""Regressão Linear para prever kills a partir do histórico do jogador."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.dataset_final import FEATURE_COLUMNS, carregar_dataset_final, separar_entradas_alvo
from src.evaluation.metricas import metricas_treino_validacao
from src.evaluation.resultados import formatar_tabela, salvar_tabela
from src.preprocessing.separacao_dados import dividir_por_tempo
from src.utils.caminhos import PASTA_METRICAS

MODELO_LINEAR = "regressao_linear"
MODELO_RIDGE = "ridge"

# Força da regularização do Ridge. Valores maiores encolhem mais os coeficientes.
ALFAS = [0.01, 0.1, 1, 10, 100, 1000, 10000]


def criar_modelo(alfa: float | None = None) -> Pipeline:
    """Regressão Linear com features padronizadas.

    Sem alfa usa mínimos quadrados; com alfa usa Ridge. O StandardScaler fica
    dentro do pipeline, então a média e o desvio são calculados só no treino.
    """
    regressor = LinearRegression() if alfa is None else Ridge(alpha=alfa)
    return make_pipeline(StandardScaler(), regressor)


def avaliar_modelos(
    treino: pd.DataFrame, validacao: pd.DataFrame, alfas: list[float] | None = None
) -> pd.DataFrame:
    """Treina a Regressão Linear e o Ridge com cada alfa e avalia no treino e na validação.

    O conjunto de teste não é usado: ele fica reservado para a avaliação final do projeto.
    """
    alfas = ALFAS if alfas is None else alfas
    X_treino, y_treino = separar_entradas_alvo(treino)
    X_val, y_val = separar_entradas_alvo(validacao)

    resultados = []
    for alfa in [None, *alfas]:
        modelo = criar_modelo(alfa).fit(X_treino, y_treino)
        resultados.append({
            "modelo": MODELO_LINEAR if alfa is None else MODELO_RIDGE,
            "alfa": 0.0 if alfa is None else alfa,  # a Regressão Linear equivale a alfa 0
            **metricas_treino_validacao(modelo, X_treino, y_treino, X_val, y_val),
        })
    return pd.DataFrame(resultados)


def resumo_validacao(avaliacao: pd.DataFrame, partidas: int) -> pd.DataFrame:
    """Uma linha para a Regressão Linear e outra para o melhor Ridge (menor MAE na validação)."""
    linear = avaliacao[avaliacao["modelo"] == MODELO_LINEAR]
    ridge = avaliacao[avaliacao["modelo"] == MODELO_RIDGE].nsmallest(1, "MAE_validacao")
    resumo = pd.concat([linear, ridge])[
        ["modelo", "alfa", "MAE_validacao", "RMSE_validacao", "R2_validacao"]
    ]
    resumo = resumo.rename(
        columns={"MAE_validacao": "MAE", "RMSE_validacao": "RMSE", "R2_validacao": "R2"}
    )
    resumo.insert(0, "conjunto", "validacao")
    resumo.insert(3, "partidas", partidas)
    return resumo.reset_index(drop=True)


def coeficientes(treino: pd.DataFrame, alfa: float | None = None) -> pd.DataFrame:
    """Coeficientes do modelo treinado, do maior para o menor em valor absoluto.

    Como as features são padronizadas, cada coeficiente é a variação em kills
    quando a feature aumenta um desvio padrão e as demais ficam fixas.
    """
    X_treino, y_treino = separar_entradas_alvo(treino)
    modelo = criar_modelo(alfa).fit(X_treino, y_treino)
    coef = modelo[-1].coef_
    return (
        pd.DataFrame({"feature": list(FEATURE_COLUMNS), "coeficiente": coef, "abs": np.abs(coef)})
        .sort_values("abs", ascending=False)
        .drop(columns="abs")
        .reset_index(drop=True)
    )


def correlacao_com_alvo(treino: pd.DataFrame) -> pd.Series:
    """Correlação de cada feature com as kills, para comparar com o sinal dos coeficientes."""
    X_treino, y_treino = separar_entradas_alvo(treino)
    return X_treino.corrwith(y_treino)


def tabela_coeficientes(treino: pd.DataFrame, alfa_ridge: float) -> pd.DataFrame:
    """Coeficientes da Regressão Linear ao lado da correlação com as kills e dos coeficientes do Ridge."""
    tabela = coeficientes(treino)
    ridge = coeficientes(treino, alfa_ridge).set_index("feature")["coeficiente"]
    tabela["correlacao_com_kills"] = tabela["feature"].map(correlacao_com_alvo(treino))
    tabela["coeficiente_ridge"] = tabela["feature"].map(ridge)
    return tabela


def main() -> None:
    # Lê data/processed/tb_lobby_stats_player_final.csv (gerado por python -m src.data.dataset_final)
    treino, validacao, _ = dividir_por_tempo(carregar_dataset_final())

    avaliacao = avaliar_modelos(treino, validacao)
    resumo = resumo_validacao(avaliacao, len(validacao))
    melhor_alfa = resumo.loc[resumo["modelo"] == MODELO_RIDGE, "alfa"].iloc[0]
    coef = tabela_coeficientes(treino, melhor_alfa)

    salvar_tabela(avaliacao, "regressao_linear_busca.csv")
    salvar_tabela(resumo, "regressao_linear_kills.csv")
    salvar_tabela(coef, "regressao_linear_coeficientes.csv")

    print(formatar_tabela(avaliacao))
    print()
    print(formatar_tabela(resumo))
    print()
    print(formatar_tabela(coef))
    print(f"Resultados salvos em {PASTA_METRICAS}")


if __name__ == "__main__":
    main()
