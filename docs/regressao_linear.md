# Regressão Linear

> Issue #15: Implementar e avaliar Regressão Linear

## 1. O que foi feito

Foi implementada uma Regressão Linear (scikit-learn) para prever `qtKill` na próxima partida do jogador, usando as 25 features de histórico do dataset final.

- **Código:** `src/models/regressao_linear.py`
- **Testes:** `tests/test_regressao_linear.py`
- **Dados:** dataset final (`data/processed/tb_lobby_stats_player_final.csv`, 172.773 linhas), dividido com `dividir_por_tempo`:
  - treino: 120.941 linhas;
  - validação: 25.916 linhas.
- **Conjunto de teste:** não foi usado. Ele fica reservado para a avaliação final, depois que todos os modelos forem comparados.

O modelo é um `Pipeline` com `StandardScaler` seguido de `LinearRegression`. A padronização deixa todas as features na mesma escala, o que permite comparar os coeficientes entre si. Como fica dentro do pipeline, a média e o desvio usados na padronização vêm só do treino.

Como as features são muito correlacionadas entre si, também foi testado o **Ridge**, a versão regularizada da regressão linear sugerida em `docs/modelos_regressao.md`. Foram testados os valores de `alpha` 0,01, 0,1, 1, 10, 100, 1000 e 10000.

Para reproduzir, com o CSV original em `data/raw/`:

```bash
python -m src.data.dataset_final
python -m src.models.regressao_linear
```

## 2. Resultados

### Validação (25.916 partidas)

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline: média histórica de kills | 5,411 | 6,974 | 0,147 |
| **Regressão Linear** | **5,362** | **6,925** | **0,159** |

O MAE no treino foi 5,313, muito próximo do da validação. Isso indica que o modelo não está sobreajustado.

### Ridge

| `alpha` | MAE validação |
|---:|---:|
| 0 (Regressão Linear) | 5,3620 |
| 0,01 a 1 | 5,3620 |
| 10 | 5,3622 |
| 100 | 5,3627 |
| 1000 | 5,3633 |
| 10000 | 5,3658 |

O Ridge não melhorou o resultado. Com 120 mil partidas de treino e só 25 features, a regressão linear comum já é estável, e a regularização só piora o erro quando fica muito forte. A tabela completa está em `results/metrics/regressao_linear_busca.csv`.

### Coeficientes (features padronizadas)

Cada coeficiente indica quantas kills a previsão muda quando a feature aumenta um desvio padrão e as outras ficam fixas.

| Feature | Coeficiente | Correlação com kills |
|---|---:|---:|
| `mediaKills_20` | −2,82 | +0,35 |
| `killsPorRound_20` | +2,63 | +0,36 |
| `mediaKills_hist` | +1,31 | +0,36 |
| `mediaDano_20` | +1,00 | +0,35 |
| `mediaRounds_20` | +0,67 | +0,07 |
| `mediaDano_5` | +0,59 | +0,29 |
| `nivelAnterior` | +0,41 | +0,25 |

A lista completa está em `results/metrics/regressao_linear_coeficientes.csv`.

## 3. Análise

- **A Regressão Linear supera o baseline, mas por pouco.** Ela erra em média 0,049 kill a menos que a média histórica (cerca de 0,9%), e o R² sobe de 0,147 para 0,159. É o melhor resultado até agora. A Árvore de Decisão da Issue #18 teve MAE 5,393.
- **Não há overfitting.** O erro no treino (5,313) e na validação (5,362) são parecidos. Um modelo linear com 25 coeficientes é simples demais para decorar 120 mil partidas.
- **Os coeficientes não devem ser lidos isoladamente.** `mediaKills_20` tem coeficiente negativo (−2,82), apesar de ter correlação positiva com as kills (+0,35). Isso acontece por **multicolinearidade**: `mediaKills_20` tem correlação de 0,98 com `mediaDano_20`, de 0,94 com `killsPorRound_20` e de 0,92 com `mediaKills_hist`. O modelo divide o efeito entre essas features, e os sinais se compensam. Em conjunto, a mensagem é a mesma do baseline: jogadores que matam mais no histórico tendem a matar mais na próxima partida.
- **Interpretação:** a relação entre o histórico e as kills da próxima partida é em grande parte linear. O ganho sobre o baseline vem de combinar várias médias, como kills, dano, rounds e nível, em vez de usar só a média de kills.

## 4. Próximos passos

- Testar a **Regressão de Poisson**, também prevista em `docs/modelos_regressao.md`. Ela é parecida com a linear, mas própria para contagens.
- Comparar com **Random Forest** e **Gradient Boosting** para ver se modelos não lineares conseguem superar o MAE de 5,362.
- Se a multicolinearidade atrapalhar a interpretação, avaliar a remoção de features redundantes (por exemplo, manter só uma entre `mediaKills_20` e `mediaDano_20`).
- A avaliação no conjunto de teste será feita só no final, com o modelo escolhido.
