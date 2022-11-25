# from IPython.display import Audio
# from gtts import gTTS as g
# def play_after_line(my_string):
#     tts = g(my_string)
#     tts.save('1.wav')
#     file = '1.wav'
#     Audio(file, autoplay=True)

import pyttsx3
def play_after_line(my_string):
    engine = pyttsx3.init()
    engine.say(my_string)
    engine.runAndWait()

# play_after_line("Hey there!")
