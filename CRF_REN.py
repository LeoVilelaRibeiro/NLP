# -*- coding: utf-8 -*-
"""
Exemplo adaptado especialmente para distribuição durante o Seminário Brasil Digital
Professor Leonardo Vilela Ribeiro
leonardo.leovilela@gmail.com
Equipe:
    Igor Chagas Marques
    Luan Lisboa
"""
import pycrfsuite # biblioteca CRF Python
import os
import pandas as pd

#setando o diretório de trabalho
os.chdir("C:/xxx/yyy")  #Change working directory

produtos_tageados = pd.read_excel('AMOSTRAFRANGO_TREINO.xlsx')
array_idx_produtos = produtos_tageados.iloc[:,:].values
#Indices:
    #0 idx
    #1 classificacao
    # palavra do produto
    
#array_idx_produtos[0]

#for idx, linha in enumerate(array_idx_produtos):
#   print(linha[1], linha[2])

# MONTANDO O DOCUMENTO DA FORMA QUE O ALGORITMO PRECISA RECEBER,

docs_geral = []
array_produto = []

for idx, linha in enumerate(array_idx_produtos):
    #print(linha)
    tupla = (linha[2], linha[1])
    if idx == 0:
        array_produto.append(tupla)
    elif(linha[0] == array_idx_produtos[idx-1][0]):
        array_produto.append(tupla)
    elif(linha[0] != array_idx_produtos[idx-1][0]):
        
        docs_geral.append(array_produto)
        array_produto = []
        array_produto.append(tupla)
        
# Último produto do loop não tem id diferente do de cima, mas precisa ser inserido no docs_geral
docs_geral.append(array_produto)

# FUNÇÃO DO CÓDIGO QUE CARACTERIZA AS FEATURES POR ORDEM E TIPO QUE APARECEM NO TEXTO

def word2features(doc, i):
    word = doc[i][0]
   

    # A Primeira configuração de features é geral, para todas as palavras
    features = [
        'bias',
        'word.lower=' + str(word).lower(),
        'word.isupper=%s' % str(word).isupper(),
        'word.istitle=%s' % str(word).istitle(),
        'word.isdigit=%s' % str(word).isdigit(),
        
    ]

    # Estas features são criada para as palavras que NAO estão no inicio do documento
    if i > 0:
        word1 = doc[i-1][0]
        
        features.extend([
            '-1:word.lower=' + str(word1).lower(),
            '-1:word.istitle=%s' % str(word1).istitle(),
            '-1:word.isupper=%s' % str(word1).isupper(),
            '-1:word.isdigit=%s' % str(word1).isdigit(),
            
        ])
    else:
        # Indica que é o início do documento
        features.append('BOS')

    # Estas features são criada para as palavras que NAO estão no fim do documento
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        #postag1 = doc[i+1][1]
        features.extend([
            '+1:word.lower=' + str(word1).lower(),
            '+1:word.istitle=%s' % str(word1).istitle(),
            '+1:word.isupper=%s' % str(word1).isupper(),
            '+1:word.isdigit=%s' % str(word1).isdigit(),
            #'+1:postag=' + postag1
        ])
    else:
        # Indica que é o fim do documento
        features.append('EOS')

    return features

####################### utilização de biblioteca para treinamento e validação:

from sklearn.model_selection import train_test_split

# A function for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    #return [label for (token, postag, label) in doc]
    return [label for (token, label) in doc]

# CARREGA AS FEATURES E AS LABELS
X = [extract_features(doc) for doc in docs_geral]
y = [get_labels(doc) for doc in docs_geral]

#MONTA A BASE DE TREINO HOLD OUT VALIDATION DE 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X[0]

trainer = pycrfsuite.Trainer(verbose=True)

# Submit training data to the trainer
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

# Set the parameters of the model
trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,  

    # maximum number of iterations
    'max_iterations': 200,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})

#configure um modelo de treino e salve-o, após isso você poderá usar o modelo treinado e salvo para executar o
#taggeamento em outras bases
trainer.train('crf.model')

tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]


# Let's take a look at a random sample in the testing set
i = 5
#len(y_pred)
#y_test[0]
for i in range(len(y_pred)):
    for  z, x, y, in zip(y_test[i], y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):
        #print("%s (%s)" % (y, x))
        print(y, x, z)
        


#X = [extract_features(doc) for doc in docs_geral]


#################BKP
