"""Métricas de regressão usadas para comparar todos os modelos do projeto.

Todos os modelos e a baseline passam por `calcular_metricas`, para que os
números das tabelas sejam sempre calculados da mesma forma.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calcular_metricas(y_real, y_previsto) -> dict[str, float]:
    """MAE e RMSE (em kills) e R² das previsões.

    MAE é o erro absoluto médio; o RMSE dá mais peso a erros grandes; o R² é a
    fração da variação das kills explicada pelas previsões (1 é o máximo).
    """
    return {
        "MAE": float(mean_absolute_error(y_real, y_previsto)),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_previsto))),
        "R2": float(r2_score(y_real, y_previsto)),
    }


def metricas_treino_validacao(modelo, X_treino, y_treino, X_val, y_val) -> dict[str, float]:
    """Erro de um modelo já treinado no treino e na validação.

    Comparar o MAE dos dois conjuntos mostra o overfitting: erro baixo no treino
    e alto na validação indica que o modelo decorou as partidas do treino.
    """
    treino = calcular_metricas(y_treino, modelo.predict(X_treino))
    validacao = calcular_metricas(y_val, modelo.predict(X_val))
    return {
        "MAE_treino": treino["MAE"],
        "MAE_validacao": validacao["MAE"],
        "RMSE_validacao": validacao["RMSE"],
        "R2_validacao": validacao["R2"],
    }
