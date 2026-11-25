import numpy as np
import pandas as pd
import pytest

from src.data.dataset_final import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation.resultados import salvar_tabela
from src.models.hist_gradient_boosting import (
    BASELINE,
    PARAMETROS,
    avaliar_no_teste,
    configuracao_selecionada,
)
from src.models.hist_gradient_boosting_pso import (
    LIMITES,
    MODELO,
    POPULACAO_MINIMA,
    AvaliadorHGB,
    buscar_hiperparametros,
    grafico_convergencia,
    inicializar_posicoes,
    otimizar_pso,
    resumo_validacao,
    vetor_para_configuracao,
)


def _conjunto(n, semente):
    rng = np.random.default_rng(semente)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    df[TARGET_COLUMN] = (20 + 5 * df["mediaKills_hist"] + rng.normal(size=n)).round()
    return df


def _esfera(posicoes):
    """Soma dos quadrados: mínimo 0 na origem."""
    return (posicoes**2).sum(axis=1)


LIMITES_ESFERA = np.array([[-5.0, 5.0]] * 6)

# Limites pequenos para os testes com o modelo rodarem rápido.
LIMITES_RAPIDOS = {**LIMITES, "max_iter": (5, 20), "max_leaf_nodes": (3, 7)}


def test_vetor_vira_configuracao_com_inteiros_arredondados_e_loss_mapeada():
    configuracao = vetor_para_configuracao(np.array([0.2, 0.05, 123.6, 14.4, 199.5, 0.7]))
    assert list(configuracao) == list(PARAMETROS)
    assert configuracao["loss"] == "squared_error"
    assert configuracao["max_iter"] == 124
    assert configuracao["max_leaf_nodes"] == 14
    assert isinstance(configuracao["max_iter"], int)
    assert configuracao["learning_rate"] == pytest.approx(0.05)
    assert configuracao["l2_regularization"] == pytest.approx(0.7)
    assert vetor_para_configuracao(np.array([0.5, 0.1, 100, 31, 20, 0]))["loss"] == "absolute_error"


def test_continuos_sao_arredondados_para_as_casas_gravadas_no_csv():
    configuracao = vetor_para_configuracao(np.array([1, 0.10668412345, 100, 15, 200, 1.38357549]))
    assert configuracao["learning_rate"] == 0.106684
    assert configuracao["l2_regularization"] == 1.383575


def test_configuracao_lida_do_csv_e_a_mesma_que_foi_avaliada(tmp_path):
    posicao = np.array([0.9, 0.10668412345, 230.2, 7, 245, 1.38357549])
    configuracao = vetor_para_configuracao(posicao)
    metricas = {"MAE": 5.355, "RMSE": 6.932, "R2": 0.157}
    salvar_tabela(resumo_validacao(configuracao, metricas, 100), "resumo.csv", tmp_path)
    assert configuracao_selecionada(pd.read_csv(tmp_path / "resumo.csv")) == configuracao


def test_avaliacao_no_teste_identifica_o_pso():
    treino, teste = _conjunto(300, 0), _conjunto(150, 2)
    configuracao = vetor_para_configuracao(np.array([0.9, 0.1, 10, 5, 20, 0.0]), LIMITES_RAPIDOS)
    resultado = avaliar_no_teste(treino, teste, configuracao, nome_modelo=MODELO)
    assert list(resultado["modelo"]) == [BASELINE, MODELO]
    assert (resultado["conjunto"] == "teste").all()
    assert (resultado["partidas"] == 150).all()
    assert resultado.loc[1, "max_iter"] == 10


def test_vetor_fora_dos_limites_e_trazido_para_a_borda():
    configuracao = vetor_para_configuracao(np.array([3.0, -1.0, 9999, 0, 10_000, 50]))
    assert configuracao["loss"] == "absolute_error"
    assert configuracao["learning_rate"] == LIMITES["learning_rate"][0]
    assert configuracao["max_iter"] == LIMITES["max_iter"][1]
    assert configuracao["max_leaf_nodes"] == LIMITES["max_leaf_nodes"][0]
    assert configuracao["min_samples_leaf"] == LIMITES["min_samples_leaf"][1]
    assert configuracao["l2_regularization"] == LIMITES["l2_regularization"][1]


def test_inicializacao_aleatoria_dentro_dos_limites_e_reproduzivel():
    limites = np.array(list(LIMITES.values()))
    posicoes = inicializar_posicoes(50, limites, np.random.default_rng(42))
    assert posicoes.shape == (50, len(PARAMETROS))
    assert ((posicoes >= limites[:, 0]) & (posicoes <= limites[:, 1])).all()
    assert len(np.unique(posicoes[:, 0])) == 50  # sorteadas, não fixas

    mesma_semente = inicializar_posicoes(50, limites, np.random.default_rng(42))
    outra_semente = inicializar_posicoes(50, limites, np.random.default_rng(7))
    assert np.array_equal(posicoes, mesma_semente)
    assert not np.array_equal(posicoes, outra_semente)


def test_populacao_menor_que_o_minimo_gera_erro():
    with pytest.raises(ValueError):
        otimizar_pso(_esfera, LIMITES_ESFERA, n_particulas=POPULACAO_MINIMA - 1)


def test_gbest_nunca_piora_e_termina_melhor_que_a_populacao_inicial():
    resultado = otimizar_pso(_esfera, LIMITES_ESFERA, n_iteracoes=30)
    valores = [registro["fitness_gbest"] for registro in resultado.historico]
    assert len(valores) == 31  # população inicial + 30 iterações
    assert all(depois <= antes for antes, depois in zip(valores, valores[1:]))
    assert valores[-1] < valores[0]
    assert resultado.fitness == valores[-1]
    assert resultado.fitness < 0.1  # chega perto do mínimo da esfera


def test_pso_e_reproduzivel_com_a_mesma_semente():
    primeiro = otimizar_pso(_esfera, LIMITES_ESFERA, semente=42)
    segundo = otimizar_pso(_esfera, LIMITES_ESFERA, semente=42)
    assert np.array_equal(primeiro.posicao, segundo.posicao)
    assert primeiro.fitness == segundo.fitness


def test_particulas_ficam_dentro_dos_limites():
    posicoes_avaliadas = []

    def fitness(posicoes):
        posicoes_avaliadas.append(posicoes.copy())
        return _esfera(posicoes - 4.9)  # mínimo perto da borda, para as partículas passarem dela

    otimizar_pso(fitness, LIMITES_ESFERA, n_iteracoes=20)
    todas = np.vstack(posicoes_avaliadas)
    assert ((todas >= -5.0) & (todas <= 5.0)).all()


def test_avaliador_nao_treina_duas_vezes_a_mesma_configuracao():
    avaliador = AvaliadorHGB(_conjunto(300, 0), _conjunto(100, 1), LIMITES_RAPIDOS)
    posicao = np.array([0.9, 0.1, 10.2, 5.0, 20.0, 0.0])
    quase_igual = np.array([0.8, 0.1, 9.8, 5.3, 19.6, 0.0])  # arredonda para a mesma configuração
    maes = avaliador(np.vstack([posicao, quase_igual]))
    assert maes[0] == maes[1]
    assert avaliador.treinos_acumulados == [1]


def test_busca_com_o_modelo_gera_historico_e_resumo_no_formato_padrao():
    treino, validacao = _conjunto(300, 0), _conjunto(100, 1)
    configuracao, metricas, historico = buscar_hiperparametros(
        treino, validacao, LIMITES_RAPIDOS, n_iteracoes=3
    )

    assert list(historico["iteracao"]) == [0, 1, 2, 3]
    assert historico["MAE_gbest"].is_monotonic_decreasing
    assert (historico["MAE_medio"] >= historico["MAE_gbest"] - 1e-12).all()
    assert historico["treinos"].is_monotonic_increasing
    assert historico["treinos"].iloc[-1] <= 10 * 4
    assert historico["MAE_gbest"].iloc[-1] == pytest.approx(metricas["MAE"])
    assert set(PARAMETROS) <= set(historico.columns)

    resumo = resumo_validacao(configuracao, metricas, len(validacao))
    assert list(resumo.columns[:2]) == ["conjunto", "modelo"]
    assert list(resumo.columns[-4:]) == ["partidas", "MAE", "RMSE", "R2"]
    assert resumo.loc[0, "modelo"] == MODELO
    assert resumo.loc[0, "partidas"] == 100


def test_grafico_de_convergencia_e_gravado(tmp_path):
    historico = pd.DataFrame({
        "iteracao": [0, 1, 2],
        "MAE_medio": [6.0, 5.6, 5.4],
        "MAE_gbest": [5.5, 5.4, 5.35],
    })
    caminho = tmp_path / "convergencia.png"
    grafico_convergencia(historico, caminho, mae_referencia=5.358)
    assert caminho.is_file() and caminho.stat().st_size > 0
