import pandas as pd
import numpy as np
import math
import nltk #Natural Language Tool Kit

from collections import Counter
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import LabelKFold
from sklearn import datasets

#nltk.download('stopwords')#Baixando e/ou atualizando as stopWords
#nltk.download('rslp')# Extrator da raiz das palavras( Removedor de Sufixo Lingua Portuguesa)
#nltk.download('punkt')#Ferramenta para trabalhar com pontuação
stopWords = nltk.corpus.stopwords.words("portuguese"); #recebendo palavras sem significado relevante
raiz = nltk.stem.RSLPStemmer()

classificacoes = pd.read_csv('emails.csv', encoding='utf-8')
textoPuro = classificacoes['email']
marcacoes=classificacoes['spam']

df = pd.DataFrame(textoPuro)

frases = textoPuro.str.lower();#textos em minusculo
textoPalavras = [nltk.tokenize.word_tokenize(frase) for frase in frases]#Limpando as pontuações do texto

dicionario = set() #Set é um conjunto, não permite repetição

for lista in textoPalavras:
    validas = [raiz.stem(palavra) for palavra in lista if palavra not in stopWords and len(palavra) > 3]
    dicionario.update(validas)

totalDePalavras = len(dicionario)

tuplas = zip(dicionario,range(0,totalDePalavras)) #criando tuplas com palavras ao lado de um índice

tradutor = {palavra:indice for palavra, indice in tuplas} #Criando um dicionário com índices

texto = textoPalavras[0] #Texto a ser analisado

def vetorizar_texto(texto, tradutor):
    vetor = [0] * len(tradutor) #Vetor para representar qualquer frase no  universo de palavras
    for palavra in texto:
        if len(palavra) > 0:
            if raiz.stem(palavra) in tradutor:
                posicao = tradutor[raiz.stem(palavra)]
                vetor[posicao]+=1
    
    return vetor

vetoresDeTexto = [vetorizar_texto(texto, tradutor) for texto in textoPalavras]

x = np.array(vetoresDeTexto)
y = np.array(marcacoes.tolist())


porcentagemDeTreino = 0.8

tamanhoDoTreino = math.ceil(porcentagemDeTreino * len(y))
tamanhoDaValidacao = len(y) - tamanhoDoTreino

#Dados a serem treinados
treinoDados = x[0:tamanhoDoTreino]
treinoMarcacoes = y[0:tamanhoDoTreino]

#Dados a serem validados a partir do treinamento
validacaoDados = x[tamanhoDoTreino:]
validacaoMarcacoes = y[tamanhoDoTreino:]

def fitAndPredict(nome, modelo, treinoDados, treinoMarcacoes):
    k = 4
    scores = cross_val_score(modelo, treinoDados, treinoMarcacoes,cv = k)
    taxaDeAcerto = np.mean(scores)

    msg = "Taxa de Acerto do {0}: {1}".format(nome, taxaDeAcerto)
    print(msg)

    return taxaDeAcerto

resultados = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))
resultadoOneVsRest = fitAndPredict("OneVsRest", modeloOneVsRest, treinoDados, treinoMarcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest


from sklearn.multiclass import OneVsOneClassifier
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state=0))
resultadoOneVsOne = fitAndPredict("OneVsOne", modeloOneVsOne, treinoDados, treinoMarcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fitAndPredict("MultinomialNB", modeloMultinomial, treinoDados, treinoMarcacoes)
resultados[resultadoMultinomial] = modeloMultinomial

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fitAndPredict("AdaBoost", modeloAdaBoost, treinoDados, treinoMarcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

print("Total de Palavras",len(dicionario))
