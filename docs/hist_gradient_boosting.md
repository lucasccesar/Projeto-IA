# HistGradientBoostingRegressor para regressão

> Issue #34: Implementar e avaliar HistGradientBoostingRegressor

## 1. O que foi feito

Foi implementado um `HistGradientBoostingRegressor` (scikit-learn) para prever `qtKill` na próxima partida do jogador, usando as 25 features de histórico do dataset final (`FEATURE_COLUMNS`). Foram testadas duas funções de perda: `loss="squared_error"` e `loss="absolute_error"`.

- **Código:** `src/models/hist_gradient_boosting.py`
- **Testes:** `tests/test_hist_gradient_boosting.py`
- **Dados:** dataset final (`data/processed/tb_lobby_stats_player_final.csv`, 172.773 linhas), dividido com `dividir_por_tempo`:
  - treino: 120.941 linhas;
  - validação: 25.916 linhas (as mesmas partidas usadas pela baseline, Regressão Linear, Árvore de Decisão e KNN).
- **Conjunto de teste:** não foi usado em nenhuma etapa de treino, ajuste de hiperparâmetros ou seleção. Depois de a configuração ser escolhida na validação, ela foi avaliada uma única vez no teste (seção 4), seguindo o mesmo protocolo do KNN.

### Como reproduzir

Dependências: as de `requirements.txt` (testado com scikit-learn 1.9.1 e pandas 3.0.6). Com o CSV original em `data/raw/`, execute na raiz do projeto:

```bash
python -m src.data.dataset_final
python -m src.models.hist_gradient_boosting
```

- Semente: `random_state=42` (constante `SEMENTE`). Duas execuções seguidas geram arquivos idênticos.
- Tempo: cerca de 1 a 2 minutos para as 64 configurações.
- Saídas em `results/metrics/`:
  - `hist_gradient_boosting_busca.csv`: todas as configurações avaliadas, com parâmetros, MAE no treino, MAE, RMSE e R² na validação e a diferença de MAE entre validação e treino;
  - `hist_gradient_boosting_kills.csv`: a melhor configuração de cada loss, no formato de resumo dos demais modelos (`conjunto`, `modelo`, hiperparâmetros, `partidas`, `MAE`, `RMSE`, `R2`), ordenada pelo MAE. A primeira linha é a configuração escolhida.
- A comparação com os outros modelos está em `notebooks/06_visualizacao_metricas.ipynb`.

A avaliação no teste é um comando separado, executado depois da busca. Ele não refaz a busca: lê a configuração de menor MAE em `hist_gradient_boosting_kills.csv`, treina o modelo só no treino e grava `results/metrics/hist_gradient_boosting_teste.csv` (modelo escolhido e baseline `baseline_media` nas mesmas partidas de teste, no mesmo formato de `knn_teste.csv`):

```bash
python -m src.models.hist_gradient_boosting --teste
```

### Early stopping desligado

O early stopping do scikit-learn separa uma parte **aleatória** do treino para decidir quando parar. Isso misturaria partidas de datas diferentes e quebraria a ordem temporal. Por isso ele foi desligado (`early_stopping=False`), e o número de iterações (`max_iter`) entra na grade de hiperparâmetros.

## 2. Hiperparâmetros testados

Para cada loss foram testadas todas as combinações abaixo (32 por loss, 64 no total). Cada configuração foi treinada no treino e avaliada na validação. A escolhida é a de **menor MAE na validação**.

| Hiperparâmetro | O que controla | Valores testados |
|---|---|---|
| `loss` | Erro minimizado no treino | `squared_error`, `absolute_error` |
| `learning_rate` | Peso de cada nova árvore | 0,05; 0,1 |
| `max_iter` | Número de árvores | 100, 300 |
| `max_leaf_nodes` | Tamanho máximo de cada árvore | 15, 31 |
| `min_samples_leaf` | Mínimo de partidas por folha | 20, 200 |
| `l2_regularization` | Encolhimento dos valores das folhas | 0, 1 |

`max_depth` ficou sem limite (`None`), porque o tamanho das árvores já é limitado por `max_leaf_nodes`. A grade é pequena de propósito: `learning_rate` e `max_iter` controlam o ajuste total, `max_leaf_nodes` e `min_samples_leaf` controlam a complexidade de cada árvore (na Árvore de Decisão, folhas grandes reduziram muito o overfitting), e `l2_regularization` testa se a regularização ajuda.

## 3. Resultados

### Melhor configuração de cada loss (validação, 25.916 partidas)

| Loss | `learning_rate` | `max_iter` | `max_leaf_nodes` | `min_samples_leaf` | `l2_regularization` | MAE | RMSE | R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`absolute_error`** (selecionada) | 0,05 | 300 | 15 | 200 | 0 | **5,358** | 6,937 | 0,156 |
| `squared_error` | 0,1 | 100 | 15 | 200 | 0 | 5,360 | 6,924 | 0,159 |

### Comparação com os modelos já implementados (validação, 25.916 partidas)

Todos os modelos foram avaliados nas mesmas 25.916 partidas de validação.

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| **HistGradientBoosting (`absolute_error`)** | **5,358** | 6,937 | 0,156 |
| HistGradientBoosting (`squared_error`) | 5,360 | **6,924** | **0,159** |
| Regressão Linear | 5,362 | 6,925 | 0,159 |
| KNN (K=300, manhattan) | 5,377 | 6,935 | 0,156 |
| Árvore de Decisão | 5,393 | 6,963 | 0,150 |
| Baseline: média histórica de kills | 5,411 | 6,974 | 0,147 |

### Efeito dos hiperparâmetros

MAE médio na validação de cada valor, considerando todas as combinações dos demais parâmetros:

| Hiperparâmetro | Valor | `absolute_error` | `squared_error` |
|---|---|---:|---:|
| `learning_rate` | 0,05 / 0,1 | 5,360 / 5,364 | 5,364 / 5,373 |
| `max_iter` | 100 / 300 | 5,360 / 5,364 | 5,364 / 5,372 |
| `max_leaf_nodes` | 15 / 31 | 5,361 / 5,363 | 5,365 / 5,372 |
| `min_samples_leaf` | 20 / 200 | 5,363 / 5,362 | 5,370 / 5,367 |
| `l2_regularization` | 0 / 1 | 5,362 / 5,362 | 5,368 / 5,369 |

Na validação, o MAE variou entre 5,358 e 5,374 com `absolute_error` e entre 5,360 e 5,394 com `squared_error`. A tabela completa está em `results/metrics/hist_gradient_boosting_busca.csv`.

## 4. Avaliação no conjunto de teste

A configuração selecionada na validação (`absolute_error`, `learning_rate=0.05`, `max_iter=300`, `max_leaf_nodes=15`, `min_samples_leaf=200`, `l2_regularization=0`) foi treinada só no treino (120.941 partidas) e avaliada uma única vez nas 25.916 partidas de teste. Nenhuma decisão foi tomada a partir destes números. O KNN aparece como referência porque também foi avaliado no mesmo teste; a baseline igual nos dois experimentos (MAE 5,412) confirma que as partidas são as mesmas.

| Modelo | MAE validação | MAE teste | RMSE teste | R² teste |
|---|---:|---:|---:|---:|
| Baseline: média histórica de kills | 5,411 | 5,412 | 7,030 | 0,129 |
| KNN (K=300, manhattan) | 5,377 | 5,370 | **6,974** | **0,143** |
| **HistGradientBoosting (`absolute_error`)** | **5,358** | **5,368** | 6,992 | 0,139 |

- O MAE no teste (5,368) ficou próximo do da validação (5,358), então a escolha feita na validação se manteve no teste.
- No teste, o modelo erra 0,045 kill a menos que a baseline (0,8%). Em relação ao KNN, a diferença no MAE é de só 0,002 kill, e o KNN tem RMSE e R² um pouco melhores. Os dois modelos têm, na prática, o mesmo desempenho.

Os resultados estão em `results/metrics/hist_gradient_boosting_teste.csv` e na seção de teste de `notebooks/06_visualizacao_metricas.ipynb`.

## 5. Análise

- **O ganho é muito pequeno.** A configuração escolhida erra 0,053 kill a menos que a média histórica (cerca de 1,0%) e 0,004 kill a menos que a Regressão Linear. Essa diferença para a Regressão Linear é muito pequena para afirmar que o Gradient Boosting é melhor. O R² fica no mesmo patamar (0,156 a 0,159).
- **Cada loss ganha na métrica que otimiza.** `absolute_error` minimiza o erro absoluto e tem o menor MAE. `squared_error` minimiza o erro quadrático e tem o menor RMSE e o maior R². Como o MAE é a métrica principal do projeto, a configuração selecionada usa `absolute_error`.
- **Modelos mais simples se saíram melhor.** Árvores menores (15 folhas), folhas maiores (200 partidas) e menos ajuste total (`learning_rate` 0,05 ou `max_iter` 100) deram o menor MAE. As configurações mais complexas reduzem o erro no treino (até 5,03) sem melhorar a validação, o que indica um começo de overfitting. A `l2_regularization` praticamente não mudou o resultado.
- **Interpretação:** assim como na Árvore de Decisão e na Regressão Linear, o histórico atual explica só uma parte pequena da variação das kills. O boosting não encontrou padrões não lineares relevantes além do que a Regressão Linear já captura.

## 6. Próximos passos

- Testar `loss="poisson"`, prevista em `docs/modelos_regressao.md`, já que `qtKill` é uma contagem.
- Se os próximos modelos também ficarem perto de MAE 5,36, priorizar novas features em vez de modelos mais complexos.
- Os resultados de teste não devem ser usados para ajustar este ou outros modelos; a escolha do modelo final continua sendo feita na validação.
