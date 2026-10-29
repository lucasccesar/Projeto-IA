# Seleção de modelos de regressão

> Issue #9 — Selecionar modelos de regressão para os experimentos
> Responsável: @LeticiaOMF

## 1. Contexto do problema

O objetivo é prever `qtKill` (quantidade de kills) de um jogador na **próxima partida**, usando apenas informações das partidas anteriores (features criadas em `src/preprocessing/feature_historico.py`).

Características dos dados que influenciaram a escolha dos modelos:

| Característica | Por que importa |
|---|---|
| Alvo numérico e inteiro (contagem de kills, média ≈ 19, desvio ≈ 7,5) | Problema de **regressão**; vale testar um modelo próprio para contagens. |
| ~184 mil registros | Descarta modelos que escalam mal com o número de linhas (ex.: SVR). |
| Features muito correlacionadas entre si (médias de kills em 5/20/histórico, dano, headshots) | Modelos lineares simples sofrem com multicolinearidade; regularização ajuda. |
| Features com `NaN` nas primeiras partidas de cada jogador | Alguns modelos precisam de imputação; outros lidam com `NaN` nativamente. |
| Relação fraca entre histórico e próxima partida (correlação ≈ 0,37 com a média de kills) | O ganho sobre uma referência simples tende a ser pequeno — por isso é obrigatório ter baselines. |
| O projeto já usa `scikit-learn` | Todos os modelos escolhidos estão disponíveis na biblioteca, sem dependências novas. |

## 2. Modelos pesquisados

| Modelo | Tipo | Decisão |
|---|---|---|
| Baseline: média do histórico do jogador | Referência | ✅ Selecionado |
| Regressão Linear / Ridge | Linear | ✅ Selecionado |
| Regressão de Poisson | Linear generalizado (contagem) | ✅ Selecionado |
| Random Forest | Ensemble de árvores (bagging) | ✅ Selecionado |
| Gradient Boosting (`HistGradientBoostingRegressor`) | Ensemble de árvores (boosting) | ✅ Selecionado |
| Lasso / ElasticNet | Linear regularizado | ➖ Variação opcional do Ridge |
| Árvore de Decisão isolada | Árvore | ❌ Coberta pela Random Forest |
| KNN Regressor | Baseado em instâncias | ❌ Descartado |
| SVR (Support Vector Regression) | Kernel | ❌ Descartado |
| Rede Neural (MLP) | Rede neural | ❌ Descartado nesta etapa |

## 3. Modelos selecionados

### 3.1 Baseline — média de kills do jogador

**Como funciona:** não aprende nada; a previsão é simplesmente a média de kills do jogador nas últimas 20 partidas (`mediaKills_20`). Também vale registrar a média geral do dataset (`DummyRegressor`).

**Por que será testado:** serve de régua. O notebook `02_features` já mostrou que prever a média geral dá MAE ≈ 5,81 e que a média das últimas 20 partidas reduz esse erro em ~6,5%. Qualquer modelo de ML só faz sentido se superar esses valores.

### 3.2 Regressão Linear com regularização (Ridge)

**Como funciona:** ajusta uma reta (hiperplano) `kills = b0 + b1·x1 + b2·x2 + ...`, escolhendo os pesos que minimizam o erro quadrático. O Ridge adiciona uma penalidade que impede os pesos de ficarem muito grandes, o que estabiliza o modelo quando as variáveis são correlacionadas.

**Por que será testado:**
- É o modelo mais simples e interpretável: os coeficientes mostram quanto cada feature pesa na previsão.
- As features do projeto são fortemente correlacionadas (médias em várias janelas, dano × kills), e o Ridge lida melhor com isso que a regressão linear pura.
- A relação entre média histórica e kills futuras parece aproximadamente linear, então pode ter desempenho competitivo.

**Cuidados:** exige imputar os `NaN` e padronizar as features (`StandardScaler`) dentro de um `Pipeline`.

### 3.3 Regressão de Poisson

**Como funciona:** é um modelo linear generalizado feito para variáveis de **contagem**. Em vez de prever kills diretamente como uma reta, modela o logaritmo da média esperada de kills, o que garante previsões sempre positivas e assume que a variância cresce junto com a média.

**Por que será testado:**
- `qtKill` é uma contagem (inteiro ≥ 0), exatamente o tipo de alvo para o qual o modelo foi criado.
- Permite comparar se tratar o alvo como contagem traz ganho em relação à regressão linear comum.
- Continua interpretável (coeficientes indicam efeito multiplicativo).

**Cuidados:** mesmas etapas de imputação e padronização do Ridge. A variância das kills (~56) é maior que a média (~19), então o ajuste pode não ser perfeito — isso faz parte do que queremos verificar.

### 3.4 Random Forest

**Como funciona:** treina muitas árvores de decisão, cada uma com uma amostra aleatória dos dados e das features, e faz a média das previsões. A combinação de várias árvores reduz o sobreajuste de uma árvore isolada.

**Por que será testado:**
- Captura relações **não lineares** e interações entre features (ex.: jogadores com muitas partidas e alta taxa de vitória) sem precisar criá-las manualmente.
- Não precisa de padronização e é robusto a outliers (ex.: partidas com 84 rounds ou dano extremo).
- Fornece importância das features, útil para discutir quais informações do histórico mais ajudam.

**Cuidados:** com ~184 mil linhas pode ser lento e consumir memória; limitar `n_estimators`, `max_depth` e usar `n_jobs=-1`.

### 3.5 Gradient Boosting (`HistGradientBoostingRegressor`)

**Como funciona:** também usa árvores, mas de forma **sequencial**: cada nova árvore é treinada para corrigir os erros das anteriores. A versão `Hist` agrupa os valores das features em faixas (histogramas), o que a torna muito rápida em bases grandes.

**Por que será testado:**
- Costuma ser o modelo de melhor desempenho em dados tabulares como os nossos.
- Trata `NaN` **nativamente**, o que é útil porque várias features de histórico ficam vazias nas primeiras partidas.
- É rápido o suficiente para 184 mil linhas e permite `loss="poisson"`, então pode ser testado também como modelo de contagem.

**Cuidados:** tem mais hiperparâmetros (`learning_rate`, `max_iter`, `max_leaf_nodes`); usar `early_stopping` e validação para evitar sobreajuste.

## 4. Modelos descartados (e por quê)

- **Árvore de Decisão isolada:** fácil de entender, mas tende a sobreajustar; a Random Forest e o Gradient Boosting já são versões melhoradas dela.
- **KNN Regressor:** precisa calcular distâncias até muitas amostras para cada previsão, o que fica lento com 184 mil linhas, e é sensível à escala e aos `NaN`.
- **SVR:** o custo de treinamento cresce muito com o número de amostras (aproximadamente quadrático ou pior), tornando-o inviável para o tamanho do dataset.
- **Rede Neural (MLP):** exige mais ajuste e é menos interpretável; em dados tabulares desse porte, raramente supera o Gradient Boosting. Pode ser revisitada se sobrar tempo.
- **Lasso / ElasticNet:** não serão um experimento separado, mas podem ser testados como variação do Ridge para ver se zerar alguns coeficientes ajuda.

## 5. Como os modelos serão comparados

Para que os experimentos sejam comparáveis entre si, todos devem seguir o mesmo protocolo:

1. **Divisão temporal dos dados:** treinar com as partidas mais antigas e testar com as mais recentes (por `dtCreatedAt`). Divisão aleatória misturaria passado e futuro do mesmo jogador.
2. **Mesmo conjunto de features e mesmo filtro** (`filtrar_minimo_partidas`, mínimo de 5 partidas anteriores).
3. **Métricas:**
   - **MAE** (principal): erro médio em kills, fácil de interpretar (“o modelo erra em média X kills”).
   - **RMSE:** penaliza mais os erros grandes.
   - **R²:** proporção da variação das kills explicada pelo modelo.
4. **Comparação sempre contra o baseline** da seção 3.1.
5. Resultados salvos em `results/metrics/`.

## 6. Próximas Issues sugeridas

- [ ] Definir divisão treino/teste temporal e funções de métricas em `src/evaluation/`
- [ ] Implementar baseline (média do histórico e `DummyRegressor`)
- [ ] Implementar e avaliar Ridge
- [ ] Implementar e avaliar Regressão de Poisson
- [ ] Implementar e avaliar Random Forest
- [ ] Implementar e avaliar `HistGradientBoostingRegressor` (perdas `squared_error` e `poisson`)
- [ ] Comparar resultados e escolher o modelo final

## Referências

- scikit-learn — Linear Models (Ridge, Lasso, PoissonRegressor): https://scikit-learn.org/stable/modules/linear_model.html
- scikit-learn — Ensembles (Random Forest, HistGradientBoosting): https://scikit-learn.org/stable/modules/ensemble.html
- scikit-learn — Métricas de regressão: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
- Notebooks do projeto: `01_eda_dataset.ipynb` e `02_features.ipynb`
