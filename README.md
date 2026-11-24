# Projeto de Inteligência Artificial — Nome Provisório

Template educacional para o desenvolvimento de um projeto acadêmico de Machine Learning na disciplina de Inteligência Artificial. Este repositório fornece apenas uma organização inicial: a definição do problema, a análise dos dados, os modelos e os experimentos deverão ser desenvolvidos pela equipe.

## Equipe

**Nome da equipe:** _preencher_

| Integrante | Conta no GitHub | Responsabilidade inicial |
| --- | --- | --- |
| Nome completo | `@usuario` | A definir |
| Nome completo | `@usuario` | A definir |
| Nome completo | `@usuario` | A definir |

## Descrição geral

A ideia do projeto é utilizar dados de partidas de CS da plataforma Gamers Club para analisar o desempenho dos jogadores e desenvolver uma solução de Machine Learning voltada à previsão da quantidade de kills realizadas em uma partida. O problema será abordado como uma tarefa de regressão, utilizando a variável qtKill como variável alvo.

## Definição da baseline

A baseline prevê as kills (`qtKill`) de uma partida usando **somente as partidas anteriores do mesmo jogador**. Foram comparadas quatro regras: média de todo o histórico anterior, média das últimas 20 partidas anteriores (`media_20`), mínimo e máximo de todo o histórico anterior. A regra `media_20` usa as partidas disponíveis quando o jogador ainda não tem 20. Os registros são ordenados por jogador, data e ID do lobby; a partida prevista fica fora do cálculo. As previsões são calculadas na ordem cronológica, como se cada partida fosse prevista depois de conhecidos os resultados das partidas anteriores, inclusive as anteriores no período de validação. O dataset final determina a divisão por data (70% treino, 15% validação e 15% teste), e as previsões são associadas às suas linhas por jogador e lobby. **As métricas abaixo usam exatamente as linhas de validação do dataset final**. Partidas do teste não entram nas métricas nem no histórico das partidas de validação.

| Regra | Partidas avaliadas | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Média histórica | 25.916 | 5,411 | 6,974 | 0,147 |
| Média das últimas 20 partidas | 25.916 | 5,464 | 7,040 | 0,131 |
| Mínimo histórico | 25.916 | 15,514 | 17,333 | -4,269 |
| Máximo histórico | 25.916 | 21,097 | 23,285 | -8,509 |

A **média histórica** é a baseline principal porque teve os menores erros e o maior R². A média das últimas 20 partidas também é registrada para comparação com a referência descrita no [documento de modelos](https://github.com/lucasccesar/Projeto-IA/pull/24). MAE é a média do erro absoluto em kills; RMSE também mede o erro em kills, dando mais peso a erros grandes. R² mede o desempenho em relação à previsão da média das kills na validação; valores negativos indicam desempenho pior que essa referência. Os resultados completos estão em [`results/metrics/baseline_kills.csv`](results/metrics/baseline_kills.csv). Para reproduzi-los, coloque `tb_lobby_stats_player.csv` em `data/raw/` e execute na raiz do projeto:

```bash
python -m src.models.baseline
```

O comando recria o dataset final em memória, recalcula as previsões e sobrescreve apenas o arquivo de métricas. Ele não altera o dataset original nem o CSV em `data/processed/`. Para comparar um modelo posterior com a baseline, use as mesmas linhas de validação e métricas.

## Resultados na validação

Todos os modelos usam as 25 features de histórico do dataset final e são comparados nas mesmas 25.916 partidas de validação, com as métricas calculadas por `src/evaluation/metricas.py`.

| Modelo | MAE | RMSE | R² | Código | Documentação |
| --- | ---: | ---: | ---: | --- | --- |
| Regressão Linear | 5,362 | 6,925 | 0,159 | `src/models/regressao_linear.py` | [`docs/regressao_linear.md`](docs/regressao_linear.md) |
| KNN (`K=300`, distância Manhattan) | 5,377 | 6,935 | 0,156 | `src/models/knn.py` | [`notebooks/05_knn.ipynb`](notebooks/05_knn.ipynb) |
| Árvore de Decisão (`max_depth=6`, `min_samples_leaf=50`) | 5,393 | 6,963 | 0,150 | `src/models/arvore_decisao.py` | [`docs/arvore_decisao.md`](docs/arvore_decisao.md) |
| Baseline: média histórica | 5,411 | 6,974 | 0,147 | `src/models/baseline.py` | seção acima |

Os gráficos e a comparação completa estão em [`notebooks/06_visualizacao_metricas.ipynb`](notebooks/06_visualizacao_metricas.ipynb).

## Como reproduzir

Com o ambiente configurado (seção abaixo) e `tb_lobby_stats_player.csv` em `data/raw/`, execute na raiz do projeto:

```bash
python -m src.data.dataset_final        # gera data/processed/tb_lobby_stats_player_final.csv
python -m src.models.baseline           # results/metrics/baseline_kills.csv
python -m src.models.regressao_linear   # results/metrics/regressao_linear_*.csv
python -m src.models.arvore_decisao     # results/metrics/arvore_decisao_*.csv e a figura de overfitting
python -m src.models.knn                # results/metrics/knn_busca.csv e knn_kills.csv
python -m src.models.hist_gradient_boosting           # hist_gradient_boosting_busca.csv e _kills.csv
python -m src.models.hist_gradient_boosting --teste   # hist_gradient_boosting_teste.csv
pytest                                  # testes automatizados
```

O fluxo dos dados é sempre o mesmo: CSV original → limpeza (`data_cleaning`) → features do histórico (`feature_historico`) → dataset final (`dataset_final`) → divisão por data (`separacao_dados`) → modelos → métricas (`evaluation`). A avaliação do KNN no conjunto de teste (`results/metrics/knn_teste.csv`) é feita em `notebooks/05_knn.ipynb`.

## Estrutura do repositório

```text
.
├── .github/                      # modelos de Issue e de Pull Request
├── data/
│   ├── raw/                      # CSVs originais (não versionados)
│   ├── processed/                # dataset final gerado (não versionado)
│   └── README.md
├── docs/
│   ├── modelos_regressao.md      # seleção dos modelos
│   ├── regressao_linear.md
│   └── arvore_decisao.md
├── notebooks/
│   ├── 00_compreensao_dataset.ipynb
│   ├── 01_eda_dataset.ipynb
│   ├── 02_features.ipynb
│   ├── 03_separacao_dados.ipynb
│   ├── 04_limpeza_dados.ipynb
│   ├── 05_knn.ipynb
│   └── 06_visualizacao_metricas.ipynb
├── src/
│   ├── data/
│   │   ├── colunas.py            # colunas que identificam a linha, a data e o alvo
│   │   └── dataset_final.py      # prepara, salva e carrega o dataset final
│   ├── preprocessing/
│   │   ├── data_cleaning.py      # limpeza e validação do dataset bruto
│   │   ├── feature_historico.py  # features das partidas anteriores do jogador
│   │   └── separacao_dados.py    # divisão por data em treino, validação e teste
│   ├── models/
│   │   ├── baseline.py
│   │   ├── regressao_linear.py
│   │   ├── arvore_decisao.py
│   │   └── knn.py
│   ├── evaluation/
│   │   ├── metricas.py           # MAE, RMSE e R², iguais para todos os modelos
│   │   └── resultados.py         # gravação e exibição das tabelas de resultados
│   └── utils/
│       └── caminhos.py           # caminhos padrão de dados e resultados
├── tests/                        # um arquivo de teste por módulo de src/
├── results/
│   ├── figures/                  # figuras geradas pelos experimentos
│   └── metrics/                  # CSVs de métricas gerados pelos experimentos
├── .gitignore
├── pyproject.toml                # configuração do pytest
├── README.md
└── requirements.txt
```

### Finalidade dos diretórios

- `data/raw/`: datasets originais, sem alterações. Os arquivos de dados não devem ser versionados por padrão.
- `data/processed/`: dados gerados por limpeza, transformação ou preparação. Esses arquivos também não devem ser versionados por padrão.
- `docs/`: documentos de decisões do projeto, como a seleção dos modelos de regressão.
- `notebooks/`: análises exploratórias, experimentos e registros de investigação em Jupyter.
- `src/data/`: carregamento dos dados e geração do dataset final de treinamento.
- `src/preprocessing/`: limpeza, criação das features do histórico e divisão temporal.
- `src/models/`: baseline e modelos. Cada módulo pode ser executado com `python -m src.models.<nome>`.
- `src/evaluation/`: métricas e gravação dos resultados, compartilhadas por todos os modelos.
- `src/utils/`: caminhos padrão e funções auxiliares compartilhadas pelo projeto.
- `tests/`: testes automatizados do código produzido pela equipe.
- `results/figures/`: figuras e gráficos gerados pelos experimentos (os PNG são versionados).
- `results/metrics/`: resultados de métricas gerados pelos experimentos (os CSV são versionados).
- `.github/`: modelos para Issues e Pull Requests.

Os notebooks devem apoiar a exploração e a comunicação dos experimentos. Coloque código reutilizável em `src/` para evitar concentrar toda a implementação nos notebooks.

### Padrões do código

- **Modelos (`src/models/`):** cada módulo começa com as constantes (`MODELO`, nomes dos arquivos de saída em `ARQUIVO_*` e hiperparâmetros testados), define `criar_modelo`, faz a busca de hiperparâmetros só na validação e termina com `main()`, executado por `python -m src.models.<nome>`.
- **Resultados:** o resumo de validação de cada modelo tem as colunas `conjunto`, `modelo`, os hiperparâmetros escolhidos, `partidas`, `MAE`, `RMSE` e `R2`. As métricas são sempre calculadas por `calcular_metricas` e as tabelas gravadas por `salvar_tabela`, com seis casas decimais.
- **Colunas e caminhos:** nomes de colunas compartilhados ficam em `src/data/colunas.py` e caminhos de pastas e arquivos em `src/utils/caminhos.py`; não repita esses valores nos módulos.
- **Estilo:** aspas duplas, linhas de até 100 caracteres, type hints nas funções públicas e docstrings em português.
- **Testes:** os testes de um módulo ficam em `tests/test_<modulo>.py`, com o mesmo nome do módulo testado.

## Configuração do ambiente

É recomendado utilizar Python 3.10 ou uma versão mais recente compatível com as dependências do projeto.

1. Clone o repositório e acesse sua pasta:

   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DO_REPOSITORIO>
   ```

2. Crie um ambiente virtual:

   ```bash
   python -m venv .venv
   ```

3. Ative o ambiente virtual:

   No Linux ou macOS:

   ```bash
   source .venv/bin/activate
   ```

   No Windows (PowerShell):

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

4. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

## Organização dos dados

Armazene os datasets originais em `data/raw/` e preserve-os sem modificações. Salve em `data/processed/` apenas os dados resultantes de limpeza, transformação ou preparação.

Datasets podem ser grandes ou conter informações que não devem ser publicadas. Por isso, o `.gitignore` impede o versionamento desses arquivos por padrão. Dos resultados, apenas as tabelas de métricas (CSV) e as figuras (PNG) são versionadas. Os arquivos `.gitkeep` mantêm a estrutura das pastas no Git. Se a equipe precisar compartilhar dados, deve combinar um meio apropriado e documentar como obtê-los em `data/README.md`.

## Fluxo de desenvolvimento

O fluxo esperado é:

```text
Issue → Branch → Commits → Pull Request → Code Review → Approval → Merge
```

Regras do projeto:

1. Toda alteração de código deve estar associada a uma Issue.
2. Cada atividade deve ser desenvolvida em uma branch própria.
3. Não devem ser realizados commits diretamente na `main`.
4. Toda alteração deve entrar na `main` por meio de Pull Request.
5. O Pull Request deve estar associado à Issue correspondente. Quando o merge concluir a atividade, use `Closes #numero_da_issue` na descrição do PR.
6. O autor do Pull Request não pode aprovar o próprio PR.
7. Todo PR precisa de pelo menos uma aprovação de outro integrante da equipe antes do merge.
8. Cada integrante deve utilizar sua própria conta GitHub e realizar seus próprios commits.
9. O histórico do GitHub será utilizado para acompanhar a participação individual dos integrantes.

### Exemplo de início de uma atividade

Depois de criar ou escolher uma Issue, atualize a `main` local e crie uma branch com um nome descritivo:

```bash
git switch main
git pull
git switch -c issue-12-descricao-curta
```

Faça commits pequenos e claros. Ao concluir a atividade, envie a branch e abra um Pull Request utilizando o modelo do repositório.

## Observação

Este template não contém uma solução de Machine Learning. A escolha das técnicas, a implementação, os testes, os experimentos e a análise dos resultados são responsabilidades dos alunos.
