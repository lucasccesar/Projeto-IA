"""Prepara, salva e carrega o dataset final de treinamento a partir das partidas brutas."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.colunas import CONTEXT_COLUMNS, DATE_COLUMN, KEY_COLUMNS, TARGET_COLUMN
from src.preprocessing.data_cleaning import clean_dataset, remove_missing_model_records
from src.preprocessing.feature_historico import (
    FEATURE_COLUMNS,
    MIN_PARTIDAS,
    criar_features_historico,
    filtrar_minimo_partidas,
)
from src.utils.caminhos import CAMINHO_DADOS_BRUTOS, CAMINHO_DATASET_FINAL

FINAL_COLUMNS = (*CONTEXT_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN)


def carregar_dados_brutos(caminho=CAMINHO_DADOS_BRUTOS) -> pd.DataFrame:
    """Lê o CSV original das partidas, sem nenhum tratamento."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Dataset original não encontrado em {caminho}. "
            "Coloque tb_lobby_stats_player.csv em data/raw/."
        )
    return pd.read_csv(caminho)


def preparar_dataset_final(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna uma linha por partida elegível, com histórico anterior e alvo atual.

    As colunas de contexto identificam a linha e permitem a divisão temporal;
    somente FEATURE_COLUMNS devem ser usadas como entradas do modelo.
    """
    limpo = clean_dataset(df)
    if limpo[list(CONTEXT_COLUMNS)].isna().any().any():
        raise ValueError("Há identificadores ou datas ausentes no dataset original.")

    historico = criar_features_historico(limpo)
    elegiveis = filtrar_minimo_partidas(historico)
    completos = remove_missing_model_records(
        elegiveis, input_columns=FEATURE_COLUMNS, target_column=TARGET_COLUMN
    )
    final = completos.loc[:, list(FINAL_COLUMNS)].reset_index(drop=True).copy()
    validar_dataset_final(final)
    return final


def validar_dataset_final(df: pd.DataFrame) -> None:
    """Verifica esquema, ausências e inconsistências que impedem o treinamento."""
    if tuple(df.columns) != FINAL_COLUMNS:
        raise ValueError("O dataset final não tem as colunas esperadas.")
    if df.empty:
        raise ValueError("O dataset final está vazio.")
    if df.isna().any().any():
        raise ValueError("O dataset final contém valores ausentes.")
    if df.duplicated(subset=list(KEY_COLUMNS)).any():
        raise ValueError("O dataset final contém partidas duplicadas para um jogador.")
    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COLUMN]):
        raise ValueError("A data do dataset final deve ser datetime.")
    if (df["qtdAnteriores"] < MIN_PARTIDAS).any():
        raise ValueError("O dataset final contém jogadores sem histórico suficiente.")
    valores = df[[*FEATURE_COLUMNS, TARGET_COLUMN]].to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        raise ValueError("O dataset final contém entradas ou alvos não finitos.")
    if (df[TARGET_COLUMN] < 0).any():
        raise ValueError("O dataset final contém kills negativas.")


def gerar_dataset_final(
    entrada=CAMINHO_DADOS_BRUTOS, saida=CAMINHO_DATASET_FINAL
) -> pd.DataFrame:
    """Executa a preparação novamente e grava o CSV no caminho informado."""
    final = preparar_dataset_final(carregar_dados_brutos(entrada))
    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(saida, index=False)
    return final


def carregar_dataset_final(caminho=None) -> pd.DataFrame:
    """Lê o CSV final com a data convertida e valida o resultado carregado."""
    caminho = CAMINHO_DATASET_FINAL if caminho is None else Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Dataset final não encontrado em {caminho}. "
            "Gere-o com: python -m src.data.dataset_final"
        )
    final = pd.read_csv(caminho, parse_dates=[DATE_COLUMN])
    validar_dataset_final(final)
    return final


def separar_entradas_alvo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Retorna as features de entrada (X) e as kills (y) de um conjunto do dataset final."""
    return df[list(FEATURE_COLUMNS)], df[TARGET_COLUMN]


def main() -> None:
    final = gerar_dataset_final()
    print(f"Dataset salvo em {CAMINHO_DATASET_FINAL}")
    print(f"Registros: {len(final)} | Colunas: {len(final.columns)}")
    print(f"Entradas: {len(FEATURE_COLUMNS)} | Alvo: {TARGET_COLUMN}")
    print(f"Valores ausentes: {int(final.isna().sum().sum())}")


if __name__ == "__main__":
    main()
