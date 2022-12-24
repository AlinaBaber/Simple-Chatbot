import speech_recognition as sr
#import google
import playsound  # to play an audio file
from gtts import gTTS  # google text to speech
import webbrowser
import wikipedia
import time
import numpy
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

from colorama import Fore, Style
import pickle
import random
import json
import datetime
from pandas.io.json import json_normalize



def there_exists(terms):
    for term in terms:
        if term in Query:
            return True
def Hello():

    tst = speak("hello sir I am your desktop assistant. ")
def record_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        # Hello()
        speak('Listening')

            # seconds of non-speaking audio before
            # a phrase is considered complete
            ##        r.pause_threshold = 0.7
            ##        audio = r.listen(source)
        audio = r.listen(source)  # listen for the audio via source
        print("Done Listening")

        Query = ''
        try:
            print("Recognizing")

                # for Listening the command in indian
            Query = r.recognize_google(audio, language='ar-SA')
            print("You said >> ", Query)  # print what user said
        except sr.UnknownValueError:  # error: recognizer does not understand
            speak('I did not get that')
        except sr.RequestError:
            speak('Say that again sir')  # error: recognizer is not connected
            # print(">>", Query.lower())
        return Query.lower()
def speak(audio_string):
    audio_string = str(audio_string)
    tts = gTTS(text=audio_string, lang='en')  # text to speech(voice)
    r = random.randint(1, 20000000)
    audio_file = 'audio' + str(r) + '.mp3'
    tts.save(audio_file)  # save as mp3
    playsound.playsound(audio_file)  # play the audio fileFore.LIGHTBLUE_EX + "User: "
    # print(f"Bot>> {audio_string}")  # print what app said
    print(Fore.LIGHTBLUE_EX + "Bot>> ",audio_string)  # print what app said
    os.remove(audio_file)  # remove audio file
def respond(Query):

    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
            responses.append(intent['responses'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    if there_exists(Query):
        greet = random.choice(responses)
        grt = random.choice(greet)
        speak(grt)
with open('intents1.json') as file:
    data = json.load(file)
df = json_normalize(data, 'intents')
df.to_csv("output.csv", index=False, sep='\t', encoding="utf-8")

training_sentences = []
training_labels = []
labels = []
responses = []
Query=''
for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

num_classes = len(labels)
# Then we use “LabelEncoder()” function provided by scikit-learn to convert the target labels into a model understandable form.
#nddfshdf
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)
vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
model = Sequential()

model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
try:
  # load trained model
    model = keras.models.load_model('chat_model')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
except:
  model.summary()
  epochs = 500
  history = model.fit(padded_sequences, numpy.array(training_labels), epochs=epochs)
  model.save("chat_model")
   # to save the fitted tokenizer
  with open('tokenizer.pickle', 'wb') as handle:
      pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# to save the fitted label encoder
  with open('label_encoder.pickle', 'wb') as ecn_file:
       pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
def weather():
    print("HI weather ")
def chat():
    # parameters
    max_len = 20

    while True:
        print(Fore.LIGHTBLUE_EX + "Your Bot Is Here " + Style.RESET_ALL)
        inpp = record_audio().lower()
        if "quit" in inpp:
            speak("Bye Take Care")
            break
        if "definition" in inpp:
            speak("Checking the wikipedia ")
            inpp = inpp.replace("search", "")

            # it will give the summary of 4 lines from
            # wikipedia we can increase and decrease
            # it also.
            result = wikipedia.summary(inpp, sentences=1)
            speak("According to wikipedia")
            print(result)
            speak(result)
            continue
        if "search" in inpp:
            search_term = inpp.replace("search", "")
            url = "https://google.com/search?q=" + search_term
            webbrowser.get().open(url)
            # result = google.summary(inpp, sentences=4)
            # speak("According to wikipedia")
            # speak(result)
            speak("Here is what I found for" + search_term + "on google")

            continue
        elif inpp == "open google":
            speak("Opening Google ")
            webbrowser.open("www.google.com")
            continue
        elif "what it is time" in inpp:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"the time is {strTime}")
            continue

        elif'open youtube' in inpp:
            webbrowser.open("youtube.com")
            speak("youtube is opened")
            continue
        elif 'wikipedia' in inpp:
            speak("searching in wikipedia")
            inpp = inpp.replace("wikipedia", "")
            results = wikipedia.summary(inpp, sentences=2)
            speak("According to wikipedia")
            print(results)
            speak(results)
        elif "play" in inpp:
            search_term = inpp.replace("play", "")
            url = "https://youtube.com/search?q=" + search_term
            webbrowser.get().open(url)



        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inpp]),
                                                                          truncating='post', maxlen=max_len))

        tag = lbl_encoder.inverse_transform([numpy.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                rr=numpy.random.choice(i['responses'])
                speak(rr)

print(Fore.YELLOW + "Press 1 for voice and 2 to type" + Style.RESET_ALL)
a=int(input())
if a == 1:
    print(Fore.YELLOW + "Start talking with the bot (say quit to stop)!" + Style.RESET_ALL)
    chat()
elif a == 2:
    print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
    import simplechatbot
    simplechatbot.chat1()
