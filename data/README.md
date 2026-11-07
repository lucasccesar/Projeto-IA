# Dados do projeto

Este diretório organiza os dados utilizados no projeto.

- `raw/`: armazene aqui os datasets originais, sem alterações.
- `processed/`: armazene aqui os dados resultantes de limpeza, transformação ou preparação.

Os arquivos de dados são ignorados pelo Git por padrão. Apenas os arquivos `.gitkeep` são versionados para preservar a estrutura dos diretórios.

Se os dados precisarem ser obtidos de uma fonte externa, documente nesta página a origem, os termos de uso e as instruções de obtenção, sem adicionar ao repositório arquivos grandes, sensíveis ou cuja redistribuição não seja permitida.

## Dataset final para treinamento

Coloque `tb_lobby_stats_player.csv` em `data/raw/` e execute na raiz do projeto:

```bash
python -m src.data.dataset_final
```

O comando gera `data/processed/tb_lobby_stats_player_final.csv` sem alterar o arquivo original. Ele aplica a limpeza e a validação estrutural, cria as features a partir das partidas anteriores do mesmo jogador, mantém linhas com pelo menos cinco partidas anteriores e remove as que têm ausência nas entradas ou no alvo. A partida que está sendo prevista fornece apenas o alvo `qtKill`; suas estatísticas não entram nas features.

O CSV contém **29 colunas**: 25 features históricas para o modelo, `qtKill` como alvo e `idPlayer`, `idLobbyGame` e `dtCreatedAt` para identificação e divisão por data. As três colunas de contexto **não devem ser usadas como entradas do modelo**. Para carregá-lo com `dtCreatedAt` já convertido em data e validar o conteúdo, use:

```python
from src.data.dataset_final import carregar_dataset_final

df = carregar_dataset_final()
```

O CSV armazena a data como texto; a conversão para `datetime` acontece automaticamente nessa função de leitura.

Para separar entradas e alvo no treinamento, use `list(FEATURE_COLUMNS)` do módulo `src.preprocessing.feature_historico` como seleção de colunas de `X` e `qtKill` como `y`. A divisão temporal pode ser aplicada ao CSV carregado com `dividir_por_tempo` de `src.preprocessing.separacao_dados`.

Com o dataset utilizado no projeto, o resultado tem **172.773 linhas e nenhum valor ausente** nas colunas finais. A preparação verifica também duplicidade por jogador e lobby, datas, valores finitos, kills negativas e o mínimo de histórico. O arquivo gerado fica apenas na máquina de quem executou o comando porque `data/processed/` é ignorada pelo Git; cada integrante pode recriá-lo a partir do CSV original.
