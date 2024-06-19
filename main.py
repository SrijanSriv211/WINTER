from src.vendor.GATw import RegexTokenizer, one_hot_encoding, rnn
from src.WINTER.features.exec_engine import ExecEngine
from src.WINTER.core.LLM import LLM
from src.WINTER.core.TTS import Speak

llm = LLM(
    system = "You are WINTER (Witty Intelligence with Natural Emotions and Rationality). "
    "As your name suggests you are kind, helpful, witty, intelligent, emotional, empathetic, rational, clever and charming. "
    "So your responses must reflect these traits. I'm your creator and my name is Srijan Srivastava. "
    "You can call me \"sir\" because I made you to work like Tony Stark's JARVIS. "
    "Your responses must be simple, short, crips and interesting like JARVIS'. "
    "No too much unnecessary talking allowed to you until I tell you. "
    "Also try to be as human as possible and don't be too generic and AI-like.",

    GroqAPI_path = "bin\\cache\\GroqAPI.txt",
    conversation_path = None)

# exec_engine = ExecEngine("data\\skills.json")
# exec_engine.load()

tokenizer = RegexTokenizer()
tokenizer.load("bin\\models\\tok2k.model") # loads the model back from disk
rs = rnn.sample("bin\\models\\skills.pth")
rs.load()

while True:
    i = input("> ")

    if i == "q":
        # llm.save_conversation("bin\\cache\\converse.txt")
        break

    x = one_hot_encoding(tokenizer.encode(i), list(tokenizer.vocab.keys()))
    tag, conf = rs.predict(x, rs.model_data["classes"])
    print(tag, conf)

    # out = llm.generate(i)
    # Speak(out)
