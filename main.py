from src.tokenizers.bpe import RegexTokenizer
from src.shared.utils import prepare_data
import json

enc = RegexTokenizer()
enc.load("bin\\cl4k.model")

with open("data\\clis\\clis.json", "r", encoding="utf-8") as f:
    obj = json.load(f)

classes = []
xy = [] # x = pattern, y = tag

for intent in obj["clis"]:
    skill = intent["skill"]
    patterns = intent["patterns"]

    if patterns == [""]:
        continue

    classes.append(skill)

    for pattern in patterns:
        tokenized_pattern = enc.encode(pattern, allowed_special="all")
        tokenized_pattern.extend([4279] * (64 - len(tokenized_pattern)))
        xy.append((tokenized_pattern, skill))

# print(len(max([x for x, y in xy], key=len)))
# print(len(classes))

import torch
data = [(torch.tensor(x, dtype=torch.long), torch.tensor(classes.index(y), dtype=torch.long)) for x, y in xy]

prepare_data(data, "data\\clis", 1, False)

"""
from src.shared.utils import dprint
from src.core.llm import GROQ

llm = GROQ(
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

while True:
    try:
        i = input("> ")

        if i.strip() == "":
            continue

        elif i == "exit":
            llm.save_conversation("bin\\cache\\converse.txt")
            break

        out = llm.generate(i)
        dprint(out)

    except Exception as e:
        llm.save_conversation("bin\\cache\\converse.txt")
        break
"""
