# Speech Processing Engines
# https://github.com/rany2/edge-tts.git
from ..shared.utils import get_bin_path, dprint
from colorama import Style, Fore, init
import playsound, asyncio, edge_tts, os
import speech_recognition as sr

init(True)

class TTS:
    def __init__(self, voice="en-US-GuyNeural"):
        self.voice = voice
        cache_path = os.path.join(get_bin_path(), "cache")
        self.output = os.path.join(cache_path, "audio.mp3")
        if not os.path.isdir(cache_path):
            os.mkdir(cache_path)

    # speak out loud the text
    def Speak(self, text):
        self.TextToSpeech(text)
        dprint(text)
        playsound.playsound(self.output)
        os.remove(self.output)

    def TextToSpeech(self, text):
        async def _main(text, voice):
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(self.output)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_main(text, self.voice))

class ASR:
    def __init__(self, phrase_time_limit = None, energy_threshold = 300, dynamic_energy_threshold = True):
        # We use SpeechRecognizer to record our audio
        # because it has a nice feature where it can detect when speech ends.
        self.recorder = sr.Recognizer()

        # recorder.energy_threshold = 1000 # change accordingly.
        self.recorder.energy_threshold = energy_threshold # change accordingly.
        # definitely do this, dynamic energy compensation lowers the energy threshold dramatically
        # to a point where the SpeechRecognizer never stops recording.
        self.recorder.dynamic_energy_threshold = dynamic_energy_threshold
        # listen for only 2 seconds
        self.phrase_time_limit = phrase_time_limit

    # listen to the microphone and return a speech to text
    def Listen(self):
        print("> ", end="")
        output = self.Recognize()
        print(output) if output.strip() != "" else None # print the output if it is not empty
        return output

    def Recognize(self):
        try:
            with sr.Microphone(sample_rate=16000) as source:
                self.recorder.adjust_for_ambient_noise(source)
                audio = self.recorder.listen(source, phrase_time_limit=self.phrase_time_limit)

            #TODO: maybe use OpenAI's whisper in future but for now stick with Google.
            return self.recorder.recognize_google(audio, language='en-in')

        except Exception as e:
            print(f"{Style.BRIGHT}{Fore.RED}{e}")
        return ""