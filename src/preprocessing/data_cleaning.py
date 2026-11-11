"""Limpeza e validação estrutural do dataset bruto.

A limpeza apenas converte tipos: não preenche valores ausentes, não remove
linhas e não remove valores extremos. A validação separa dois tipos de problema:

- bloqueios (`blocking_issues`): impedem o uso dos dados e geram erro em `clean_dataset`;
- sinais para revisão (`review_flags`): relações suspeitas entre colunas, que são
  apenas contadas porque podem ser partidas legítimas.
"""

import pandas as pd

from src.data.colunas import DATE_COLUMN, KEY_COLUMNS
from src.preprocessing.feature_historico import REQUIRED_FEATURE_COLUMNS

INTEGER_COLUMNS = (
    "idLobbyGame",
    "idPlayer",
    "idRoom",
    "qtKill",
    "qtAssist",
    "qtDeath",
    "qtHs",
    "qtBombeDefuse",
    "qtBombePlant",
    "qtTk",
    "qtTkAssist",
    "qt1Kill",
    "qt2Kill",
    "qt3Kill",
    "qt4Kill",
    "qt5Kill",
    "qtPlusKill",
    "qtFirstKill",
    "qtHits",
    "qtShots",
    "qtLastAlive",
    "qtClutchWon",
    "qtRoundsPlayed",
    "vlLevel",
    "qtSurvived",
    "qtTrade",
    "qtFlashAssist",
    "qtHitHeadshot",
    "qtHitChest",
    "qtHitStomach",
    "qtHitLeftAtm",  # o nome vem assim do dataset original (seria LeftArm)
    "qtHitRightArm",
    "qtHitLeftLeg",
    "qtHitRightLeg",
    "flWinner",
)
NUMERIC_COLUMNS = ("vlDamage",)
DATE_COLUMNS = (DATE_COLUMN,)
NON_NEGATIVE_COLUMNS = INTEGER_COLUMNS + NUMERIC_COLUMNS
WINNER_COLUMN = "flWinner"

# (nome do sinal, coluna que não deveria ser maior, coluna de referência)
REVIEW_RELATIONS = (
    ("headshots_exceed_kills", "qtHs", "qtKill"),
    ("deaths_exceed_rounds", "qtDeath", "qtRoundsPlayed"),
    ("hits_exceed_shots", "qtHits", "qtShots"),
    ("headshot_hits_exceed_hits", "qtHitHeadshot", "qtHits"),
    ("kills_exceed_shots", "qtKill", "qtShots"),
)

# Inconsistências que bloqueiam o uso dos dados, na ordem em que são relatadas.
BLOCKING_INCONSISTENCIES = (
    "negative_values",
    "invalid_numeric_values",
    "invalid_dates",
    "invalid_winner",
    "duplicate_rows",
    "duplicate_player_lobby",
)
BLOCKING_ISSUE_LABELS = {
    "negative_values": "valores negativos",
    "invalid_numeric_values": "valores numéricos inválidos",
    "invalid_dates": "datas inválidas",
    "invalid_winner": "valores inválidos em flWinner",
    "duplicate_rows": "linhas duplicadas",
    "duplicate_player_lobby": "duplicidades na combinação idPlayer e idLobbyGame",
    "date_columns_not_datetime": "colunas de data sem tipo datetime",
    "missing_required_columns": "colunas obrigatórias ausentes",
}


def _ensure_dataframe(df) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("A entrada deve ser um DataFrame.")


def _to_numeric(series: pd.Series) -> pd.Series:
    """Converte para número; o que não for conversível vira nulo."""
    return pd.to_numeric(series, errors="coerce")


def _to_datetime(series: pd.Series) -> pd.Series:
    """Converte para data; o que não for conversível vira nulo."""
    return pd.to_datetime(series, errors="coerce")


def _invalid_mask(source: pd.Series, converted: pd.Series) -> pd.Series:
    """Marca os valores preenchidos que viraram nulo na conversão, ou seja, os inválidos."""
    return source.notna() & converted.isna()


def _to_valid_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Converte a coluna para número e falha se houver valor não numérico."""
    numeric = _to_numeric(df[column])
    if _invalid_mask(df[column], numeric).any():
        raise ValueError(f"A coluna {column} contém valores que não são numéricos.")
    return numeric


def identify_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela com a quantidade e a proporção de valores ausentes em cada coluna."""
    _ensure_dataframe(df)
    total_rows = len(df)
    missing_count = df.isna().sum()
    return pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.astype("int64").to_numpy(),
        "missing_rate": (missing_count / total_rows if total_rows else missing_count).to_numpy(),
    })


def remove_missing_model_records(
    df: pd.DataFrame, input_columns, target_column: str
) -> pd.DataFrame:
    """Remove as linhas com valor ausente em alguma entrada do modelo ou no alvo.

    Só as colunas selecionadas são consideradas; ausências nas demais são mantidas.
    """
    _ensure_dataframe(df)
    if isinstance(input_columns, str):
        input_columns = [input_columns]
    selected_columns = list(dict.fromkeys([*input_columns, target_column]))
    missing_columns = [column for column in selected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Colunas selecionadas ausentes: {', '.join(missing_columns)}.")
    return df.dropna(subset=selected_columns).copy(deep=True)


def identify_duplicates(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    """Retorna todas as linhas envolvidas em duplicidade (linha inteira ou nas colunas de `subset`)."""
    _ensure_dataframe(df)
    duplicate_mask = df.duplicated(subset=subset, keep=False)
    return df.loc[duplicate_mask].copy(deep=True)


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Converte contagens e identificadores para inteiro anulável (Int64) e o dano para número.

    Valores ausentes são preservados. Falha se houver texto em coluna numérica
    ou valor fracionário em coluna inteira.
    """
    _ensure_dataframe(df)
    converted = df.copy(deep=True)

    for column in INTEGER_COLUMNS:
        if column not in converted.columns:
            continue
        numeric = _to_valid_numeric(converted, column)
        if (numeric.notna() & numeric.mod(1).ne(0)).any():
            raise ValueError(f"A coluna {column} deve conter apenas valores inteiros.")
        converted[column] = numeric.astype("Int64")

    for column in NUMERIC_COLUMNS:
        if column in converted.columns:
            converted[column] = _to_valid_numeric(converted, column)

    return converted


def convert_dates(df: pd.DataFrame, columns=DATE_COLUMNS) -> pd.DataFrame:
    """Converte as colunas de data para datetime e falha se houver data inválida."""
    _ensure_dataframe(df)
    converted = df.copy(deep=True)

    for column in columns:
        if column not in converted.columns:
            continue
        parsed = _to_datetime(converted[column])
        if _invalid_mask(converted[column], parsed).any():
            raise ValueError(f"A coluna {column} contém datas inválidas.")
        converted[column] = parsed

    return converted


def _relation_count(df: pd.DataFrame, left_column: str, right_column: str) -> int:
    """Quantidade de linhas em que `left_column` é maior que `right_column`."""
    if left_column not in df.columns or right_column not in df.columns:
        return 0
    left = _to_numeric(df[left_column])
    right = _to_numeric(df[right_column])
    return int((left > right).fillna(False).sum())


def detect_inconsistencies(df: pd.DataFrame) -> dict:
    """Conta as inconsistências do dataset, sem alterá-lo.

    Retorna um dicionário com contagens por coluna (valores negativos, números e
    datas inválidos) e contagens gerais (flWinner fora de 0/1, duplicatas e as
    relações suspeitas de REVIEW_RELATIONS).
    """
    _ensure_dataframe(df)

    negative_values = {}
    invalid_numeric_values = {}
    for column in NON_NEGATIVE_COLUMNS:
        if column not in df.columns:
            continue
        numeric = _to_numeric(df[column])
        invalid_mask = _invalid_mask(df[column], numeric)
        if invalid_mask.any():
            invalid_numeric_values[column] = int(invalid_mask.sum())
        negative_mask = (numeric < 0).fillna(False)
        if negative_mask.any():
            negative_values[column] = int(negative_mask.sum())

    invalid_dates = {}
    for column in DATE_COLUMNS:
        if column not in df.columns:
            continue
        invalid_mask = _invalid_mask(df[column], _to_datetime(df[column]))
        if invalid_mask.any():
            invalid_dates[column] = int(invalid_mask.sum())

    invalid_winner = 0
    if WINNER_COLUMN in df.columns:
        winner = _to_numeric(df[WINNER_COLUMN])
        invalid_winner = int((df[WINNER_COLUMN].notna() & ~winner.isin([0, 1])).sum())

    duplicate_player_lobby = 0
    if all(column in df.columns for column in KEY_COLUMNS):
        duplicate_player_lobby = len(identify_duplicates(df, subset=list(KEY_COLUMNS)))

    inconsistencies = {
        "negative_values": negative_values,
        "invalid_numeric_values": invalid_numeric_values,
        "invalid_dates": invalid_dates,
        "invalid_winner": invalid_winner,
        "duplicate_rows": len(identify_duplicates(df)),
        "duplicate_player_lobby": duplicate_player_lobby,
    }
    for name, left_column, right_column in REVIEW_RELATIONS:
        inconsistencies[name] = _relation_count(df, left_column, right_column)

    return inconsistencies


def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as conversões de tipo e de data, sem validar o resultado."""
    return convert_dates(convert_data_types(df))


def validate_cleaned_dataset(df: pd.DataFrame) -> dict:
    """Valida o dataset limpo e separa bloqueios de sinais para revisão.

    Retorna um dicionário com `is_valid`, `blocking_issues`, `review_flags`,
    `missing_values` (tabela de ausências), `rows` e `columns`.
    """
    _ensure_dataframe(df)
    inconsistencies = detect_inconsistencies(df)
    blocking_issues = {}

    missing_required_columns = [
        column for column in REQUIRED_FEATURE_COLUMNS if column not in df.columns
    ]
    if missing_required_columns:
        blocking_issues["missing_required_columns"] = missing_required_columns

    for name in BLOCKING_INCONSISTENCIES:
        if inconsistencies[name]:
            blocking_issues[name] = inconsistencies[name]

    date_type_issues = [
        column
        for column in DATE_COLUMNS
        if column in df.columns and not pd.api.types.is_datetime64_any_dtype(df[column])
    ]
    if date_type_issues:
        blocking_issues["date_columns_not_datetime"] = date_type_issues

    review_flags = {
        name: inconsistencies[name]
        for name, _, _ in REVIEW_RELATIONS
        if inconsistencies[name]
    }

    return {
        "is_valid": not blocking_issues,
        "blocking_issues": blocking_issues,
        "review_flags": review_flags,
        "missing_values": identify_missing_values(df),
        "rows": len(df),
        "columns": len(df.columns),
    }


def _format_blocking_issues(blocking_issues: dict) -> str:
    """Monta a mensagem de erro com os bloqueios encontrados."""
    details = []
    for name, value in blocking_issues.items():
        label = BLOCKING_ISSUE_LABELS.get(name, name)
        if isinstance(value, dict):
            description = ", ".join(f"{column}={count}" for column, count in value.items())
        elif isinstance(value, list):
            description = ", ".join(str(item) for item in value)
        else:
            description = str(value)
        details.append(f"{label} ({description})")
    return "A limpeza encontrou inconsistências críticas: " + "; ".join(details) + "."


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna uma cópia limpa do dataset bruto, ou falha se houver bloqueio.

    O DataFrame recebido não é alterado.
    """
    _ensure_dataframe(df)
    cleaned = apply_cleaning(df)
    validation = validate_cleaned_dataset(cleaned)
    if validation["blocking_issues"]:
        raise ValueError(_format_blocking_issues(validation["blocking_issues"]))
    return cleaned
