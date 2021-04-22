import tkinter as tk
from PIL import Image, ImageTk

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model

model = load_model('chatbot_model.h5')
import json
import random

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", tk.END)

    if msg != '':
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, "You: \n" + msg + '\n')
        ChatLog.config(foreground="#161616", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(tk.END, "Bot: \n" + res + '\n')

        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)


print("ChatBot Project is running ...")
root = tk.Tk()
root.configure(bg='#484444')

# set window title :
root.title("Interview ChatBot Project")
root.resizable(width=tk.FALSE, height=tk.FALSE)

canvas = tk.Canvas(root, width=500, height=610)
canvas.grid(columnspan=2, rowspan=2)

# logo setting :
logo = Image.open('logo.png')
logo = ImageTk.PhotoImage(logo)
logo_label = tk.Label(image=logo)
logo_label.image = logo

ChatLog = tk.Text(root, bd=0, bg="white", height="8", width="50", font="Arial")
ChatLog.config(state=tk.DISABLED)

scrollbar = tk.Scrollbar(root, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set

# Button setting
SendButton = tk.Button(root, font=("Arial, sans serif", 12, 'bold'), text="Send", width="9", height=3,
                       bd=0, bg="#1e81b0", activebackground="#33a5db", fg='#ffffff',
                       command=send)

# Create the box to enter message
EntryBox = tk.Text(root, bd=0, bg="white", width="29", height="5", font="Arial")

text = tk.Label(root, text="Copyright © - All right reserved by Mohamed Wahbi Yaakoub 2021")

ChatLog.config(state=tk.NORMAL)
ChatLog.config(foreground="#161616", font=("Arial", 12))
ChatLog.insert(tk.END, "Bot: Hello ! \nToday we are going to start your interview" + '\n')

# Place all components on the screen :
ChatLog.place(x=6, y=225, height=300, width=475)
SendButton.place(x=380, y=530, height=50)
EntryBox.place(x=6, y=530, height=50, width=370)
scrollbar.place(x=485, y=225, height=300)
logo_label.place(x=0, y=0)
text.place(x=70, y=585)
root.mainloop()
print("Bye Bye Ms ChatBot ...")
