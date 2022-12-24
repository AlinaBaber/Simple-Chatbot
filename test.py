# for Listening the command in indian
import speech_recognition as sr
r = sr.Recognizer()
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)
    # Hello()
    print('Listening')

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
        print('I did not get that')
    except sr.RequestError:
        print('Say that again sir')  # error: recognizer is not connected
        # print(">>", Query.lower())
print(Query.lower())