import sqlite3 as sq
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

from tensorflow import keras
from keras.models import Model,Sequential
from keras.layers import Dense,TextVectorization,Embedding, LSTM, Bidirectional

db_stack = sq.connect("./NLP/db_stack.db")
cursor = db_stack.cursor()

cursor.execute("SELECT * FROM Stackoverflow")
results = cursor.fetchall()
'''for r in results:
    print(results)*/'''
#Création d'un DataFrame à partir des données récupérées
df = pd.read_sql_query("SELECT * FROM Stackoverflow", db_stack)

vecteur = TextVectorization(max_tokens=150_000,output_sequence_length=1300,output_mode='int')
# le vecteur apprend mtn notre vocabulaire 
vecteur.adapt(df['Questions'].values)
# Il faut mtn vectorisé nos tweets
text_vectorise = vecteur(df['Questions'].values) 

# On cree le dataset avec les tweets vectorisés et les labels (on ajoute des options supplémentaire qui vont servire a l'entrainement)
dataset = tf.data.Dataset.from_tensor_slices((text_vectorise, df['Questions'].values)).cache().shuffle(60000).batch(10).prefetch(8)
# Ici on va reperatir le contenu de notre dataset avec la partie entrainement 80%, evaluation 10% et test 10%
octante = int(len(dataset)*.8)
dix = int(len(dataset)*.1)
train,val,test = (dataset.take(octante),dataset.skip(octante).take(dix),dataset.skip(octante+dix).take(dix))



model = Sequential()
model.add(Embedding(150_001, 32))
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
             loss=tf.keras.losses.BinaryCrossentropy())
res_train = model.fit(train, epochs=5, validation_data=val)

# Sauvegarde le model 
model.save('model_sof.h5')

# On sauvegarde le vecteur
pickle.dump({'config': vecteur.get_config(),
             'weights': vecteur.get_weights()}
            , open("monVecteur.pkl", "wb"))
