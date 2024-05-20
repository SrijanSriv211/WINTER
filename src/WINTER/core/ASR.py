import speech_recognition as sr
from colorama import Style, Fore, init

# import colorama module to differentiate between speech output and error
init(True)

# We use SpeechRecognizer to record our audio
# because it has a nice feature where it can detect when speech ends.
recorder = sr.Recognizer()
recorder.energy_threshold = 1000 # change accordingly.
# Definitely do this, dynamic energy compensation lowers the energy threshold dramatically
# to a point where the SpeechRecognizer never stops recording.
recorder.dynamic_energy_threshold = False

# Listen to the microphone and return a speech to text
def Listen(phrase_time_limit=2):
    output = ""
    try:
        with sr.Microphone(sample_rate=16000) as source:
            recorder.adjust_for_ambient_noise(source)
            audio = recorder.listen(source, phrase_time_limit=phrase_time_limit)

        output = recorder.recognize_google(audio, language='en-in')

    except Exception as e:
        print(f"{Style.BRIGHT}{Fore.RED}{e}")

    return output
