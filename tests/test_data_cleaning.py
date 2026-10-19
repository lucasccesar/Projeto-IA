import pandas as pd
import pytest

from src.preprocessing.data_cleaning import (
    apply_cleaning,
    clean_dataset,
    convert_data_types,
    convert_dates,
    detect_inconsistencies,
    identify_duplicates,
    identify_missing_values,
    validate_cleaned_dataset,
)
from src.preprocessing.feature_historico import criar_features_historico


def _required_schema_dataframe():
    return pd.DataFrame({
        "idPlayer": [1, 2],
        "idLobbyGame": [10, 11],
        "dtCreatedAt": ["2022-01-01", "2022-01-02"],
        "qtKill": [3, 4],
        "vlDamage": [100.0, 200.0],
        "qtHitHeadshot": pd.array([pd.NA, 2], dtype="Int64"),
        "qtRoundsPlayed": [20, 25],
        "flWinner": [1, 0],
        "qtShots": [100, 120],
        "vlLevel": [5, 6],
    })


@pytest.mark.parametrize("column", [
    "idPlayer",
    "idLobbyGame",
    "dtCreatedAt",
    "qtKill",
    "vlDamage",
    "qtHitHeadshot",
    "qtRoundsPlayed",
    "flWinner",
    "qtShots",
    "vlLevel",
])
def test_clean_dataset_rejects_missing_required_feature_column(column):
    df = _required_schema_dataframe().drop(columns=[column])

    with pytest.raises(ValueError, match="colunas obrigatórias ausentes") as error:
        clean_dataset(df)

    assert column in str(error.value)


def test_validate_cleaned_dataset_marks_missing_required_columns_as_blocking():
    df = _required_schema_dataframe().drop(columns=["idPlayer", "qtKill"])
    df["dtCreatedAt"] = pd.to_datetime(df["dtCreatedAt"])

    validation = validate_cleaned_dataset(df)

    assert validation["is_valid"] is False
    assert validation["blocking_issues"]["missing_required_columns"] == [
        "idPlayer",
        "qtKill",
    ]


def test_clean_dataset_accepts_required_schema_with_legitimate_missing_values():
    df = _required_schema_dataframe()
    original = df.copy(deep=True)

    cleaned = clean_dataset(df)
    validation = validate_cleaned_dataset(cleaned)

    assert df.equals(original)
    assert len(cleaned) == len(df)
    assert pd.isna(cleaned.loc[0, "qtHitHeadshot"])
    assert validation["is_valid"] is True
    assert validation["blocking_issues"] == {}


def test_identify_missing_values_reports_counts_without_filling():
    df = pd.DataFrame({
        "idPlayer": [1, 2],
        "qtKill": pd.array([3, pd.NA], dtype="Int64"),
    })

    report = identify_missing_values(df)

    missing = report.set_index("column")
    assert missing.loc["qtKill", "missing_count"] == 1
    assert missing.loc["idPlayer", "missing_count"] == 0
    assert df.loc[1, "qtKill"] is pd.NA


def test_identify_duplicates_reports_complete_and_key_duplicates():
    df = pd.DataFrame({
        "idPlayer": [1, 1, 2],
        "idLobbyGame": [10, 10, 20],
        "qtKill": [3, 3, 4],
    })

    complete_duplicates = identify_duplicates(df)
    key_duplicates = identify_duplicates(df, subset=["idPlayer", "idLobbyGame"])

    assert len(complete_duplicates) == 2
    assert len(key_duplicates) == 2


def test_convert_data_types_preserves_missing_values_and_valid_values():
    df = pd.DataFrame({
        "idLobbyGame": ["10", "11"],
        "idPlayer": ["1", "2"],
        "qtKill": ["3", None],
        "vlDamage": ["12.5", "10"],
        "flWinner": ["1", "0"],
    })

    converted = convert_data_types(df)

    assert str(converted["idLobbyGame"].dtype) == "Int64"
    assert str(converted["qtKill"].dtype) == "Int64"
    assert pd.api.types.is_numeric_dtype(converted["vlDamage"])
    assert converted.loc[0, "qtKill"] == 3
    assert pd.isna(converted.loc[1, "qtKill"])
    assert df.loc[0, "qtKill"] == "3"


def test_convert_data_types_rejects_fractional_integer_values():
    df = pd.DataFrame({"qtKill": [1.5]})

    with pytest.raises(ValueError, match="inteiros"):
        convert_data_types(df)


def test_convert_dates_converts_valid_values_and_rejects_invalid_values():
    valid_df = pd.DataFrame({"dtCreatedAt": ["2022-01-01 12:00:00", None]})
    converted = convert_dates(valid_df)

    assert pd.api.types.is_datetime64_any_dtype(converted["dtCreatedAt"])
    assert pd.isna(converted.loc[1, "dtCreatedAt"])
    assert valid_df.loc[0, "dtCreatedAt"] == "2022-01-01 12:00:00"

    invalid_df = pd.DataFrame({"dtCreatedAt": ["data-invalida"]})
    with pytest.raises(ValueError, match="datas inválidas"):
        convert_dates(invalid_df)


def test_detect_inconsistencies_reports_invalid_and_semantic_values():
    df = pd.DataFrame({
        "idPlayer": [1],
        "idLobbyGame": [10],
        "dtCreatedAt": ["data-invalida"],
        "qtKill": [-1],
        "qtHs": [0],
        "qtDeath": [3],
        "qtRoundsPlayed": [2],
        "qtHits": [3],
        "qtShots": [2],
        "qtHitHeadshot": [4],
        "flWinner": [2],
    })

    inconsistencies = detect_inconsistencies(df)

    assert inconsistencies["negative_values"]["qtKill"] == 1
    assert inconsistencies["invalid_dates"]["dtCreatedAt"] == 1
    assert inconsistencies["invalid_winner"] == 1
    assert inconsistencies["headshots_exceed_kills"] == 1
    assert inconsistencies["deaths_exceed_rounds"] == 1
    assert inconsistencies["hits_exceed_shots"] == 1
    assert inconsistencies["headshot_hits_exceed_hits"] == 1


def test_clean_dataset_does_not_change_input_fill_nulls_or_remove_outliers():
    df = pd.DataFrame({
        "idLobbyGame": [10, 11],
        "idPlayer": [1, 1],
        "dtCreatedAt": ["2022-01-01", "2022-01-02"],
        "qtKill": pd.array([3, pd.NA], dtype="Int64"),
        "vlDamage": [12.5, 99999.0],
        "qtHitHeadshot": [1, 2],
        "qtRoundsPlayed": [20, 20],
        "qtShots": [100, 100],
        "vlLevel": [5, 5],
        "flWinner": [1, 0],
    })
    original = df.copy(deep=True)

    cleaned = clean_dataset(df)

    assert cleaned is not df
    assert df.equals(original)
    assert len(cleaned) == len(df)
    assert pd.isna(cleaned.loc[1, "qtKill"])
    assert cleaned.loc[1, "vlDamage"] == 99999.0
    assert pd.api.types.is_datetime64_any_dtype(cleaned["dtCreatedAt"])


def test_clean_dataset_reports_critical_validation_errors():
    df = pd.DataFrame({
        "idLobbyGame": [10],
        "idPlayer": [1],
        "dtCreatedAt": ["2022-01-01"],
        "qtKill": [-1],
        "vlDamage": [100.0],
        "qtHitHeadshot": [1],
        "qtRoundsPlayed": [20],
        "qtShots": [100],
        "vlLevel": [5],
        "flWinner": [1],
    })

    with pytest.raises(ValueError, match="inconsistências críticas.*valores negativos"):
        clean_dataset(df)


def test_apply_cleaning_is_composable():
    df = pd.DataFrame({
        "idLobbyGame": [10],
        "idPlayer": [1],
        "dtCreatedAt": ["2022-01-01"],
        "qtKill": [3],
    })

    cleaned = apply_cleaning(df)

    assert pd.api.types.is_integer_dtype(cleaned["qtKill"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["dtCreatedAt"])


def test_validation_keeps_review_flags_separate_from_blocking_issues():
    df = pd.DataFrame({
        "idLobbyGame": [10],
        "idPlayer": [1],
        "dtCreatedAt": ["2022-01-01"],
        "qtKill": [1],
        "qtHs": [2],
        "qtDeath": [3],
        "qtRoundsPlayed": [2],
        "vlDamage": [100.0],
        "qtHitHeadshot": [1],
        "qtShots": [10],
        "vlLevel": [5],
        "flWinner": [1],
    })

    validation = validate_cleaned_dataset(clean_dataset(df))

    assert validation["is_valid"] is True
    assert validation["blocking_issues"] == {}
    assert validation["review_flags"]["headshots_exceed_kills"] == 1
    assert validation["review_flags"]["deaths_exceed_rounds"] == 1


def test_clean_dataset_remains_compatible_with_historical_features():
    df = pd.DataFrame({
        "idLobbyGame": [1, 2, 3, 4, 5, 6],
        "idPlayer": [10, 10, 10, 10, 10, 10],
        "dtCreatedAt": [
            "2022-01-01", "2022-01-02", "2022-01-03",
            "2022-01-04", "2022-01-05", "2022-01-06",
        ],
        "qtKill": [10, 20, 30, 40, 50, 60],
        "qtShots": [100, 200, 300, 400, 500, 600],
        "vlDamage": [1000, 2000, 3000, 4000, 5000, 6000],
        "qtHitHeadshot": [5, 10, 15, 20, 25, 30],
        "qtRoundsPlayed": [20, 20, 20, 20, 20, 20],
        "flWinner": [1, 0, 1, 0, 1, 0],
        "vlLevel": [5, 5, 6, 6, 7, 7],
    })

    features = criar_features_historico(clean_dataset(df))

    assert len(features) == len(df)
    assert features.loc[1, "mediaKills_hist"] == 10
    assert features.loc[2, "qtdAnteriores"] == 2
