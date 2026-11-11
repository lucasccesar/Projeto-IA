"""Gravação e exibição das tabelas de resultados, com o mesmo formato em todos os modelos."""

from pathlib import Path

import pandas as pd

from src.utils.caminhos import PASTA_METRICAS

CASAS_ARQUIVO = 6
CASAS_TELA = 3


def salvar_tabela(tabela: pd.DataFrame, nome_arquivo: str, pasta=PASTA_METRICAS) -> Path:
    """Grava a tabela como CSV em results/metrics/ e retorna o caminho do arquivo."""
    caminho = Path(pasta) / nome_arquivo
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(caminho, index=False, float_format=f"%.{CASAS_ARQUIVO}f")
    return caminho


def formatar_tabela(tabela: pd.DataFrame) -> str:
    """Texto da tabela para exibir no terminal, com três casas decimais."""
    return tabela.to_string(index=False, float_format=lambda valor: f"{valor:.{CASAS_TELA}f}")
