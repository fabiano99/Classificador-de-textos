import csv
import pandas as pd
def carregar_acessos():

    dados = []
    marcacoes = []

    arquivo = open('acessos.csv', 'r')
    leitor = csv.reader(arquivo)
    next(leitor)
    for home,como_funciona,contato,comprou in leitor:

        dados.append([int(home), int(como_funciona), int(contato)])
        marcacoes.append(int(comprou))

    return dados, marcacoes

def carregar_buscas():
    X = []
    Y = []
    arquivo = open('buscas.csv','r')
    leitor = csv.reader(arquivo)
    next(leitor)

    for home,busca,logado,comprou in leitor:
        dado = [int(home), busca, int(logado)]
        X.append(dado)
        Y.append(int(comprou))

    return X, Y