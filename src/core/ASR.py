import speech_recognition as sr
from colorama import Style, Fore, init

# import colorama module to differentiate between speech output and error
init(True)

class ASR:
    def __init__(self, phrase_time_limit = None, energy_threshold = 300, dynamic_energy_threshold = True):
        # We use SpeechRecognizer to record our audio
        # because it has a nice feature where it can detect when speech ends.
        self.recorder = sr.Recognizer()

        # recorder.energy_threshold = 1000 # change accordingly.
        self.recorder.energy_threshold = energy_threshold # change accordingly.
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically
        # to a point where the SpeechRecognizer never stops recording.
        self.recorder.dynamic_energy_threshold = dynamic_energy_threshold
        # Listen for only 2 seconds
        self.phrase_time_limit = phrase_time_limit

    # Listen to the microphone and return a speech to text
    def Listen(self):
        output = ""
        print("> ", end="")
        try:
            with sr.Microphone(sample_rate=16000) as source:
                self.recorder.adjust_for_ambient_noise(source)
                audio = self.recorder.listen(source, phrase_time_limit=self.phrase_time_limit)

            #TODO: Maybe use OpenAI's whisper in future but for now stick with Google.
            output = self.recorder.recognize_google(audio, language='en-in')

        except Exception as e:
            print(f"{Style.BRIGHT}{Fore.RED}{e}")

        # Print the output if it is not empty
        print(output) if output.strip() != "" else None
        return output
