#obtendo dados para treinamento bom pagador x mal pagador
#[clt, >= 30 anos, casado, filhos, residência, veiculo]

mal1 = [0,1,1,1,0,1]
mal2 = [1,0,0,0,0,1]
mal3 = [1,0,0,1,0,0]
mal4 = [0,0,0,0,1,0]
bom1 = [1,0,1,1,0,1]
bom2 = [1,1,1,1,1,1]
bom3 = [1,0,0,0,0,1]
bom4 = [0,1,0,0,0,1]
# -1 = Mal pagador
# 1  = Bom pagador

dados = [mal1, mal2, mal3, mal4, bom1, bom2, bom3, bom4]
marcacoes = [-1,-1,-1,-1,1,1,1,1]

#dados para prever

#[clt, idade, casado, filhos, residência, veiculo]

prever1 = [1,1,1,0,1,1]#bom
prever2 = [0,0,0,0,0,0]#mal
prever3 = [1,0,0,0,0,1]#bom
prever4 = [1,1,1,1,0,1]#bom
prever5 = [1,1,0,1,0,0]#mal

#matriz resultante deverá ser: [1,-1,1,1,-1]
resultado = [prever1, prever2, prever3, prever4, prever5]

#treinando o modelo
from sklearn.naive_bayes import MultinomialNB

modelo  = MultinomialNB()
modelo.fit(dados, marcacoes)

#predição
print(modelo.predict(resultado))

