import pandas as pd

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
    "qtHitLeftAtm",
    "qtHitRightArm",
    "qtHitLeftLeg",
    "qtHitRightLeg",
    "flWinner",
)

NUMERIC_COLUMNS = ("vlDamage",)
DATE_COLUMNS = ("dtCreatedAt",)
NON_NEGATIVE_COLUMNS = INTEGER_COLUMNS + NUMERIC_COLUMNS
REVIEW_RELATIONS = (
    ("headshots_exceed_kills", "qtHs", "qtKill"),
    ("deaths_exceed_rounds", "qtDeath", "qtRoundsPlayed"),
    ("hits_exceed_shots", "qtHits", "qtShots"),
    ("headshot_hits_exceed_hits", "qtHitHeadshot", "qtHits"),
    ("kills_exceed_shots", "qtKill", "qtShots"),
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


def _ensure_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("A entrada deve ser um DataFrame.")


def _numeric_series(series):
    return pd.to_numeric(series, errors="coerce")


def identify_missing_values(df):
    _ensure_dataframe(df)
    total_rows = len(df)
    missing_count = df.isna().sum()
    return pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.astype("int64").to_numpy(),
        "missing_rate": (missing_count / total_rows if total_rows else missing_count).to_numpy(),
    })


def identify_duplicates(df, subset=None):
    _ensure_dataframe(df)
    duplicate_mask = df.duplicated(subset=subset, keep=False)
    return df.loc[duplicate_mask].copy(deep=True)


def convert_data_types(df):
    _ensure_dataframe(df)
    converted = df.copy(deep=True)

    for column in INTEGER_COLUMNS:
        if column not in converted.columns:
            continue
        source = converted[column]
        numeric = _numeric_series(source)
        invalid_mask = source.notna() & numeric.isna()
        if invalid_mask.any():
            raise ValueError(f"A coluna {column} contém valores que não são numéricos.")
        fractional_mask = numeric.notna() & numeric.mod(1).ne(0)
        if fractional_mask.any():
            raise ValueError(f"A coluna {column} deve conter apenas valores inteiros.")
        converted[column] = numeric.astype("Int64")

    for column in NUMERIC_COLUMNS:
        if column not in converted.columns:
            continue
        source = converted[column]
        numeric = _numeric_series(source)
        invalid_mask = source.notna() & numeric.isna()
        if invalid_mask.any():
            raise ValueError(f"A coluna {column} contém valores que não são numéricos.")
        converted[column] = numeric

    return converted


def convert_dates(df, columns=DATE_COLUMNS):
    _ensure_dataframe(df)
    converted = df.copy(deep=True)

    for column in columns:
        if column not in converted.columns:
            continue
        source = converted[column]
        parsed = pd.to_datetime(source, errors="coerce")
        invalid_mask = source.notna() & parsed.isna()
        if invalid_mask.any():
            raise ValueError(f"A coluna {column} contém datas inválidas.")
        converted[column] = parsed

    return converted


def _relation_count(df, left_column, right_column):
    if left_column not in df.columns or right_column not in df.columns:
        return 0
    left = _numeric_series(df[left_column])
    right = _numeric_series(df[right_column])
    relation_mask = (left > right).fillna(False)
    return int(relation_mask.sum())


def detect_inconsistencies(df):
    _ensure_dataframe(df)
    negative_values = {}
    invalid_numeric_values = {}

    for column in NON_NEGATIVE_COLUMNS:
        if column not in df.columns:
            continue
        source = df[column]
        numeric = _numeric_series(source)
        invalid_mask = source.notna() & numeric.isna()
        if invalid_mask.any():
            invalid_numeric_values[column] = int(invalid_mask.sum())
        negative_mask = (numeric < 0).fillna(False)
        if negative_mask.any():
            negative_values[column] = int(negative_mask.sum())

    invalid_dates = {}
    for column in DATE_COLUMNS:
        if column not in df.columns:
            continue
        source = df[column]
        parsed = pd.to_datetime(source, errors="coerce")
        invalid_mask = source.notna() & parsed.isna()
        if invalid_mask.any():
            invalid_dates[column] = int(invalid_mask.sum())

    invalid_winner = 0
    if "flWinner" in df.columns:
        winner = _numeric_series(df["flWinner"])
        invalid_mask = df["flWinner"].notna() & ~winner.isin([0, 1])
        invalid_winner = int(invalid_mask.sum())

    duplicate_rows = int(len(identify_duplicates(df)))
    duplicate_player_lobby = 0
    key_columns = ["idPlayer", "idLobbyGame"]
    if all(column in df.columns for column in key_columns):
        duplicate_player_lobby = int(len(identify_duplicates(df, subset=key_columns)))

    inconsistencies = {
        "negative_values": negative_values,
        "invalid_numeric_values": invalid_numeric_values,
        "invalid_dates": invalid_dates,
        "invalid_winner": invalid_winner,
        "duplicate_rows": duplicate_rows,
        "duplicate_player_lobby": duplicate_player_lobby,
    }
    for name, left_column, right_column in REVIEW_RELATIONS:
        inconsistencies[name] = _relation_count(df, left_column, right_column)

    return inconsistencies


def apply_cleaning(df):
    cleaned = convert_data_types(df)
    return convert_dates(cleaned)


def validate_cleaned_dataset(df):
    _ensure_dataframe(df)
    inconsistencies = detect_inconsistencies(df)
    blocking_issues = {}

    missing_required_columns = [
        column
        for column in REQUIRED_FEATURE_COLUMNS
        if column not in df.columns
    ]
    if missing_required_columns:
        blocking_issues["missing_required_columns"] = missing_required_columns

    for name in (
        "negative_values",
        "invalid_numeric_values",
        "invalid_dates",
    ):
        if inconsistencies[name]:
            blocking_issues[name] = inconsistencies[name]

    for name in ("invalid_winner", "duplicate_rows", "duplicate_player_lobby"):
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


def _format_blocking_issues(blocking_issues):
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


def clean_dataset(df):
    _ensure_dataframe(df)
    cleaned = apply_cleaning(df)
    validation = validate_cleaned_dataset(cleaned)
    if validation["blocking_issues"]:
        raise ValueError(_format_blocking_issues(validation["blocking_issues"]))
    return cleaned
