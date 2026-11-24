"""Caminhos padrão do projeto, calculados a partir da raiz do repositório.

Ficam em um único lugar para que os módulos e notebooks leiam e gravem sempre
nos mesmos arquivos, independentemente da pasta de onde o comando é executado.
"""

from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parents[2]

PASTA_DADOS_BRUTOS = RAIZ_PROJETO / "data" / "raw"
PASTA_DADOS_PROCESSADOS = RAIZ_PROJETO / "data" / "processed"
PASTA_METRICAS = RAIZ_PROJETO / "results" / "metrics"
PASTA_FIGURAS = RAIZ_PROJETO / "results" / "figures"

CAMINHO_DADOS_BRUTOS = PASTA_DADOS_BRUTOS / "tb_lobby_stats_player.csv"
CAMINHO_DATASET_FINAL = PASTA_DADOS_PROCESSADOS / "tb_lobby_stats_player_final.csv"
