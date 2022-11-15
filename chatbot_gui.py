import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import pandas as pd
global dflag
hrec = "Which hospital would you recommend me, which hospital should i go, which hospital is best for this disease, where should i go, which doctor do you suggest, which hospital would treat this, what to do now"
trans_df = pd.read_csv('Data/Transformed.csv')
htags = pd.read_csv('Data/Hospitals.csv')

stemmer = LancasterStemmer()

model = load_model('chatbot_ddhr.h5')
intents = json.loads(open('intentdataset.json').read())
words = pickle.load(open('words.pkl','rb'))
labels = pickle.load(open('labels.pkl','rb'))

#will clean strings in bag of words process
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

#returns bag of words from the string
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

#predicts the intent class with its probability
def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
    print(return_list)
    return return_list

#getting chatbot response
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    global dflag
    if cosine_sim(hrec, text) >= 0.2:

        rslt_df = trans_df[trans_df['Disease'] == dflag]
        rslt_df = rslt_df.groupby(['Hospital'])['Sentiment Polarity'].mean().reset_index()

        rh = rslt_df[rslt_df['Sentiment Polarity'] == max(rslt_df['Sentiment Polarity'])]['Hospital'].reset_index(
            drop=True)

        h = htags[htags['Tags'] == rh[0]]['Hospital_Name'].reset_index(drop=True)[0]

        res = 'You are recommended to visit ' +h+' for treating '+ dflag
    else:
        ints = predict_class(text, model)
        dflag = ints[0].get('intent')
        res = getResponse(ints, intents)
    return res

import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

from sklearn.feature_extraction.text import TfidfVectorizer

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


#chatbot GUI
import tkinter
from tkinter import *
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()

photo = PhotoImage(file = "icon.png")
base.iconphoto(False, photo)

base.title("RSCare Bot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial", wrap=WORD)
ChatLog.config(state=DISABLED)
#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
#Create Button to send message
SendButton = Button(base, font=("Times",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#7289e8", activebackground="#1e46eb",fg='#ffffff',
                    command= send)
#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Times", wrap=WORD)

#EntryBox.bind("<Return>", send)
#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
base.mainloop()
