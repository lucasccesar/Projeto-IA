import pandas as pd

PROP_TREINO = 0.70
PROP_VALIDACAO = 0.15
# O restante (1 - treino - validação) é o teste: 15%


def dividir_por_tempo(df, prop_treino=PROP_TREINO, prop_validacao=PROP_VALIDACAO,
                      coluna_data='dtCreatedAt'):
    """Divide o dataset em treino, validação e teste respeitando a ordem do tempo.

    As linhas mais antigas vão para o treino, as seguintes para a validação e as mais recentes para o teste. O corte é feito pela data, então linhas com a mesma data ficam sempre no mesmo conjunto.
    """
    if prop_treino <= 0 or prop_validacao <= 0 or prop_treino + prop_validacao >= 1:
        raise ValueError("As proporções de treino e validação devem somar menos que 1.")

    df = df.sort_values(coluna_data).reset_index(drop=True)
    datas = df[coluna_data]

    corte_treino = datas.quantile(prop_treino)
    corte_validacao = datas.quantile(prop_treino + prop_validacao)

    treino = df[datas <= corte_treino]
    validacao = df[(datas > corte_treino) & (datas <= corte_validacao)]
    teste = df[datas > corte_validacao]

    return treino, validacao, teste


def resumo_separacao(treino, validacao, teste, coluna_data='dtCreatedAt'):
    """Tabela com a quantidade de linhas, a porcentagem e o período de cada conjunto."""
    conjuntos = {'Treino': treino, 'Validação': validacao, 'Teste': teste}
    total = sum(len(c) for c in conjuntos.values())

    return pd.DataFrame({
        'Linhas': [len(c) for c in conjuntos.values()],
        'Porcentagem (%)': [round(len(c) / total * 100, 2) for c in conjuntos.values()],
        'Primeira data': [c[coluna_data].min() for c in conjuntos.values()],
        'Última data': [c[coluna_data].max() for c in conjuntos.values()],
        'Jogadores': [c['idPlayer'].nunique() for c in conjuntos.values()],
    }, index=list(conjuntos))


def verificar_ordem_temporal(treino, validacao, teste, coluna_data='dtCreatedAt'):
    """Garante que o treino é anterior à validação e a validação anterior ao teste."""
    assert treino[coluna_data].max() < validacao[coluna_data].min(), "Treino invade a validação"
    assert validacao[coluna_data].max() < teste[coluna_data].min(), "Validação invade o teste"
