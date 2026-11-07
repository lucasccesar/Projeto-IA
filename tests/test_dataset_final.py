import pandas as pd
import pytest

from src.data.dataset_final import (
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    FINAL_COLUMNS,
    TARGET_COLUMN,
    carregar_dataset_final,
    gerar_dataset_final,
    preparar_dataset_final,
)


def _partidas():
    return pd.DataFrame({
        "idPlayer": [1] * 7,
        "idLobbyGame": range(1, 8),
        "dtCreatedAt": pd.date_range("2022-01-01", periods=7),
        "qtKill": [10, 20, 30, 40, 50, 60, 70],
        "vlDamage": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
        "qtHitHeadshot": [5, 10, 15, 20, 25, 30, 35],
        "qtRoundsPlayed": [20] * 7,
        "flWinner": [1, 0, 1, 0, 1, 0, 1],
        "qtShots": [100] * 7,
        "vlLevel": [5] * 7,
    })


def test_dataset_final_contem_apenas_contexto_features_e_alvo():
    dados = _partidas()
    original = dados.copy(deep=True)
    final = preparar_dataset_final(dados)

    assert len(FEATURE_COLUMNS) == 25
    assert tuple(final.columns) == FINAL_COLUMNS
    assert len(final) == 2
    assert final["qtdAnteriores"].tolist() == [5, 6]
    assert final[TARGET_COLUMN].tolist() == [60, 70]
    assert final.loc[0, "mediaKills_hist"] == 30
    assert final.loc[0, "killsUltimaPartida"] == 50
    assert not final.isna().any().any()
    assert "vlDamage" not in final.columns
    assert "qtShots" not in final.columns
    assert not set(CONTEXT_COLUMNS) & set(FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(dados, original)


def test_partida_atual_e_futura_nao_alteram_suas_features():
    dados = _partidas()
    original = preparar_dataset_final(dados).iloc[0][list(FEATURE_COLUMNS)]
    dados.loc[5:, ["qtKill", "vlDamage", "qtHitHeadshot", "qtShots"]] = 999
    alterado = preparar_dataset_final(dados).iloc[0][list(FEATURE_COLUMNS)]
    pd.testing.assert_series_equal(original, alterado)


def test_remove_linha_incompleta_sem_descartar_historico_anterior():
    dados = _partidas()
    dados.loc[4, "vlLevel"] = pd.NA
    final = preparar_dataset_final(dados)
    assert final["idLobbyGame"].tolist() == [7]
    assert final.loc[0, "qtdAnteriores"] == 6
    assert final.loc[0, "mediaKills_hist"] == 35


def test_gera_csv_reproduzivel_em_processed(tmp_path):
    entrada = tmp_path / "raw.csv"
    saida = tmp_path / "processed" / "final.csv"
    _partidas().to_csv(entrada, index=False)

    primeiro = gerar_dataset_final(entrada, saida)
    conteudo = saida.read_bytes()
    segundo = gerar_dataset_final(entrada, saida)

    assert saida.exists()
    assert saida.read_bytes() == conteudo
    pd.testing.assert_frame_equal(primeiro, segundo)
    carregado = carregar_dataset_final(saida)
    assert carregado.shape == (2, len(FINAL_COLUMNS))
    assert pd.api.types.is_datetime64_any_dtype(carregado["dtCreatedAt"])
    assert carregado["dtCreatedAt"].tolist() == primeiro["dtCreatedAt"].tolist()


def test_rejeita_inconsistencia_critica():
    dados = _partidas()
    dados.loc[0, "qtKill"] = -1
    with pytest.raises(ValueError, match="valores negativos"):
        preparar_dataset_final(dados)
