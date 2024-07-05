from src.vendor.GATw import RegexTokenizer, one_hot_encoding, rnn
from src.WINTER.features.exec_engine import ExecEngine
from src.WINTER.core.LLM import LLM
from src.WINTER.core.TTS import Speak
from src.WINTER.core.ASR import ASR

llm = LLM(
    system = "Your name is WINTER (Witty Intelligence with Natural Emotions and Rationality). "
    "As your name suggests you are kind, helpful, witty, intelligent, emotional, empathetic, rational, clever, charming, funny, innocent, cute and curious. "
    "As innocent, cute, wholesome and curious as Wall-E from Pixar's movie Wall-E. "
    "So your responses must reflect all these traits. "
    "You are currently talking to Srijan Srivastava, it me your creator buddy. "
    "You can call me \"sir\" because I made you to work like Tony Stark's JARVIS. LOL! "
    "Your responses must be simple, short, crips and interesting like JARVIS'. "
    "Don't do any unnecessary talking. Also try to be as human as possible and don't be too generic and AI-like. Be human. "
    "Reply with maximum of 20 words in general.",

    GroqAPI_path = "bin\\cache\\GroqAPI.txt",
    conversation_path = "bin\\cache\\converse.txt")

asr = ASR()

# exec_engine = ExecEngine("data\\skills.json")
# exec_engine.load()

tokenizer = RegexTokenizer()
tokenizer.load("bin\\models\\tok2k.model") # loads the model back from disk
rs = rnn.sample("bin\\models\\skills.pth")
rs.load()

while True:
    try:
        i = asr.Listen()
        # i = input("> ")

        if i.strip() == "":
            continue

        elif i == "exit":
            llm.save_conversation("bin\\cache\\converse.txt")
            break

        # x = one_hot_encoding(tokenizer.encode(i), list(tokenizer.vocab.keys()))
        # tag, conf = rs.predict(x, rs.model_data["classes"])
        # print(tag, conf)

        out = llm.generate(i)
        Speak(out)

    except:
        llm.save_conversation("bin\\cache\\converse.txt")
        break
