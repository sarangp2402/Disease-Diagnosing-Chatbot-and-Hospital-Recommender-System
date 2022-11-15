import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tensorflow
import random
import json
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import numpy as np

stemmer = LancasterStemmer()

#loading dataset
with open("intentdataset.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

#tokenizing pattern on symptoms in dataset
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

#stemming words and then sorting the root words
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

#saving words and labels
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(labels,open('labels.pkl','wb'))

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

#converting input of string into bag of words
for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

#building sequential NN model
model = Sequential()
model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]), activation='softmax'))


#compiling model with accelerated nesterov stochastic gradient descent gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


#fitting model
hist = model.fit(np.array(training), np.array(output), epochs=1000, batch_size=8, verbose=1)

#saving the model
model.save('chatbot_ddhr.h5', hist)



