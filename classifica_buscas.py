import pandas as  pd
from sklearn.naive_bayes import MultinomialNB
data_frame = pd.read_csv('buscas.csv')

X_df= data_frame[['home','busca','logado']]
Y_df = data_frame['comprou']

X_dummies = pd.get_dummies(X_df)
Y_dummies = Y_df

X = X_dummies.values
Y = Y_dummies.values

#Definição das porcentagens de treino e teste
tamanho_treino =int(0.9*len(Y))
tamanho_teste = len(Y) - tamanho_treino

#Separação dos dados para treinar o modelo
treino_dados = X[:tamanho_treino]
treino_marcacoes = Y[:tamanho_treino]

#Separação dos dados que irão testar o modelo
teste_dados = X[-tamanho_teste:]
teste_marcacoes = Y[-tamanho_teste:]










def fit_and_predict(modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)

    diferencas = resultado - teste_marcacoes
    acertos = [d for d in diferencas if d == 0]
    total_de_acertos = len(acertos)
    total_de_elementos = len(teste_dados)
    taxa_de_acerto = 100.0 * total_de_acertos/total_de_elementos

    print(taxa_de_acerto)

    return taxa_de_acerto

from sklearn.naive_bayes import MultinomialNB
modelo1  = MultinomialNB()

resultadoMultinomial = fit_and_predict(modelo1, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.ensemble import AdaBoostClassifier
modelo2 = AdaBoostClassifier()

resultadoAdaBoost = fit_and_predict(modelo2, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

if(resultadoMultinomial > resultadoAdaBoost):
    print("Multinomial NB foi o vencedor")

else:
    print("AdaBoost foi o vencedor")


