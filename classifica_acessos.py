from dados import carregar_acessos

X,Y = carregar_acessos()

treino_dados= X[:14]
treino_marcacoes = Y[:14]
teste_dados = X[:-4]
teste_marcacoes = Y[:-4]

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()

modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)

diferencas = resultado - teste_marcacoes
acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(X)
taxa_de_acerto = 100.0 * total_de_acertos/total_de_elementos


print(taxa_de_acerto)