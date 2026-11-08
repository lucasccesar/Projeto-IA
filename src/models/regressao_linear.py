"""Regressão Linear para prever kills a partir do histórico do jogador."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.dataset_final import FEATURE_COLUMNS, TARGET_COLUMN, carregar_dataset_final
from src.preprocessing.separacao_dados import dividir_por_tempo


# Força da regularização do Ridge. Valores maiores encolhem mais os coeficientes.
ALFAS = [0.01, 0.1, 1, 10, 100, 1000, 10000]


def separar_entradas_alvo(df):
    """Retorna as features de entrada (X) e as kills (y)."""
    return df[list(FEATURE_COLUMNS)], df[TARGET_COLUMN]


def criar_modelo(alfa=None):
    """Regressão Linear com features padronizadas. Sem alfa usa mínimos quadrados; com alfa usa Ridge.

    O StandardScaler fica dentro do pipeline, então a média e o desvio são calculados só no treino.
    """
    regressor = LinearRegression() if alfa is None else Ridge(alpha=alfa)
    return make_pipeline(StandardScaler(), regressor)


def calcular_metricas(y_real, y_previsto):
    """MAE, RMSE e R² em kills."""
    return {
        "MAE": mean_absolute_error(y_real, y_previsto),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_previsto))),
        "R2": r2_score(y_real, y_previsto),
    }


def _nome(alfa):
    return "regressao_linear" if alfa is None else "ridge"


def avaliar_modelos(treino, validacao, alfas=None):
    """Treina a Regressão Linear e o Ridge com cada alfa e avalia no treino e na validação.

    O conjunto de teste não é usado: ele fica reservado para a avaliação final do projeto.
    """
    alfas = ALFAS if alfas is None else alfas
    X_treino, y_treino = separar_entradas_alvo(treino)
    X_val, y_val = separar_entradas_alvo(validacao)

    resultados = []
    for alfa in [None, *alfas]:
        modelo = criar_modelo(alfa).fit(X_treino, y_treino)
        m_treino = calcular_metricas(y_treino, modelo.predict(X_treino))
        m_val = calcular_metricas(y_val, modelo.predict(X_val))
        resultados.append({
            "modelo": _nome(alfa),
            "alfa": 0.0 if alfa is None else alfa,
            "MAE_treino": m_treino["MAE"],
            "MAE_validacao": m_val["MAE"],
            "RMSE_validacao": m_val["RMSE"],
            "R2_validacao": m_val["R2"],
        })
    return pd.DataFrame(resultados)


def resumo_validacao(avaliacao, partidas):
    """Uma linha para a Regressão Linear e outra para o melhor Ridge (menor MAE na validação)."""
    linear = avaliacao[avaliacao["modelo"] == "regressao_linear"]
    ridge = avaliacao[avaliacao["modelo"] == "ridge"].nsmallest(1, "MAE_validacao")
    resumo = pd.concat([linear, ridge])[["modelo", "alfa", "MAE_validacao", "RMSE_validacao", "R2_validacao"]]
    resumo = resumo.rename(columns={"MAE_validacao": "MAE", "RMSE_validacao": "RMSE", "R2_validacao": "R2"})
    resumo.insert(0, "conjunto", "validacao")
    resumo.insert(3, "partidas", partidas)
    return resumo.reset_index(drop=True)


def coeficientes(treino, alfa=None):
    """Coeficientes do modelo treinado.

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


def correlacao_com_alvo(treino):
    """Correlação de cada feature com as kills, para comparar com o sinal dos coeficientes."""
    return treino[list(FEATURE_COLUMNS)].corrwith(treino[TARGET_COLUMN])


def main():
    raiz = Path(__file__).resolve().parents[2]
    pasta = raiz / "results/metrics"
    pasta.mkdir(parents=True, exist_ok=True)

    # Lê data/processed/tb_lobby_stats_player_final.csv (gerado por python -m src.data.dataset_final)
    dataset_final = carregar_dataset_final()
    treino, validacao, _ = dividir_por_tempo(dataset_final)

    avaliacao = avaliar_modelos(treino, validacao)
    resumo = resumo_validacao(avaliacao, len(validacao))

    coef = coeficientes(treino)
    coef["correlacao_com_kills"] = coef["feature"].map(correlacao_com_alvo(treino))
    melhor_alfa = resumo.loc[resumo["modelo"] == "ridge", "alfa"].iloc[0]
    coef_ridge = coeficientes(treino, melhor_alfa).set_index("feature")["coeficiente"]
    coef["coeficiente_ridge"] = coef["feature"].map(coef_ridge)

    avaliacao.to_csv(pasta / "regressao_linear_busca.csv", index=False, float_format="%.6f")
    resumo.to_csv(pasta / "regressao_linear_kills.csv", index=False, float_format="%.6f")
    coef.to_csv(pasta / "regressao_linear_coeficientes.csv", index=False, float_format="%.6f")

    formato = lambda v: f"{v:.3f}"
    print(avaliacao.to_string(index=False, float_format=formato))
    print()
    print(resumo.to_string(index=False, float_format=formato))
    print()
    print(coef.to_string(index=False, float_format=formato))
    print(f"Resultados salvos em {pasta}")


if __name__ == "__main__":
    main()
