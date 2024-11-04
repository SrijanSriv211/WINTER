from src.tokenizers.bpe import RegexTokenizer
from src.models.rnn import sample

enc = RegexTokenizer()
enc.load("bin\\cl4k.model")

from src.shared.utils import prepare_data
from src.shared.nltk_utils import tokenize, remove_special_chars, one_hot_encoding
import json

with open("data\\clis\\clis.json", "r", encoding="utf-8") as f:
    obj = json.load(f)

# classes = []
# xy = [] # x = pattern, y = tag

# for intent in obj["clis"]:
#     skill = intent["skill"]
#     patterns = intent["patterns"]

#     classes.append(skill)

#     if patterns == [""]:
#         continue

#     for pattern in patterns:
#         tokenized_pattern = enc.encode(pattern, allowed_special="all")
#         tokenized_pattern.extend([4279] * (64 - len(tokenized_pattern)))
#         xy.append((tokenized_pattern, skill))

# # # print(len(max([x for x, y in xy], key=len)))
# # # print(len(classes))

# import torch
# data = [(torch.tensor(x, dtype=torch.long), torch.tensor(classes.index(y), dtype=torch.long)) for x, y in xy]

classes = []
vocab = []
xy = [] # x = pattern, y = tag

for intent in obj["clis"]:
    skill = intent["skill"]
    patterns = intent["patterns"]

    if skill != "default" and patterns == [""]:
        continue

    classes.append(skill)

    for pattern in patterns:
        tokenized_words = tokenize(pattern)
        vocab.extend(tokenized_words)
        xy.append((pattern, skill))

# lemmatize, lower each word and remove unnecessary chars.
vocab = remove_special_chars(vocab)

# remove duplicates and sort
vocab = sorted(set(vocab))
classes = sorted(set(classes))

# import torch
# # create dataset for training
# data = [(torch.tensor(one_hot_encoding(x, vocab), dtype=torch.long), torch.tensor(classes.index(y), dtype=torch.long)) for x, y in xy]

# prepare_data(data, "data\\clis", 1, False)

# import pickle
# with open("data\\clis\\classes.bin", "wb") as f:
#     pickle.dump(classes, f)

test = [
    ("WINTER play me some BBS videos.", "play,on_youtube"),
    ("open Google chrome and Can you please search google for me?", "search,google"),
    ("What is a game engine?", "search,google"),
    ("please search on google What is a game engine?", "search,google"),
    ("Google Chrome", "open,app"),
    ("Please start Unity game engine for me please", "open,app"),
    ("open youtube.com please", "open,website"),
    ("if you don't mind would you please search google about shahrukh khan", "search,google"),
    ("how to start a youtube channel", "search,google"),
    ("what is the capital of paris", "search,google"),
    ("please search on google what is the capital of paris", "search,google"),
    ("reload this PC, it needs some freshness", "restart,PC"),
    ("i'm gonna be away for a while, please lock this pc", "sleep,PC"),
    ("shutdown my workstation please", "shutdown,PC"),
    ("shutdown google chrome", "shutdown,app"),
    ("kill this app", "shutdown,app"),
    ("tell me current date please", "date,current"),
    ("tell me today's time", "time,current"),
    ("is it monday today?", "day,current"),
    ("Please start Unity game engine for me please", "open,app"),
    ("start chrome.exe", "open,app"),
    ("How do you make a game engine and remember to make it in c++", "search,google"),
    ("open chrome and search on the internet How do you make a game engine", "search,google"),
    ("please make the volume one hundred percent.", "volume,specific_level"),
    ("turn the volume to zero percent.", "volume,mute"),
    ("I think that things won't be any perfect than a quite PC", "volume,mute"),
    ("I think that things won't be any perfect than a quite PC. Mute this device please.", "volume,mute"),
    ("I don't think that things won't be any perfect than a quite PC.", "volume,unmute"),
    ("set volume one hundred percent.", "volume,specific_level"),
    ("set the volume to zero percent.", "volume,mute"),
    ("mute my pc for a minute", "volume,mute"),
    ("unmute my PC now.", "volume,unmute"),
    ("increase the volume of this PC.", "volume,up"),
    ("please increase the volume", "volume,up"),
    ("please increase the volume by ten percent.", "volume,up"),
    ("please reduce the volume by fifteen points.", "volume,down"),
    ("set volume fifty nine", "volume,specific_level"),
    ("pronto, I want you to set the volume slider to sixty nine.", "volume,specific_level"),
    ("aalkjsdflkja9ierfasdfjlkaj adjfkkladjsfkl", "default"),
    ("You know WINTER, my system has two button. One for sleep and the other one is for lock. No matter what you choose, they will perform the same function. Try it if you want.", "sleep,PC"),
    ("Lock this PC", "sleep,PC"),
    ("ya would you please put my system on sleep I'm going away for some time actually.", "sleep,PC"),
    ("please put my PC a lock. I won't be here.", "sleep,PC"),
    ("I think you should now send this PC to a quite deep sleep.", "sleep,PC"),
    ("Go to sleep WINTER.", "sleep,WINTER"),
    ("Sleep WINTER", "sleep,WINTER"),
    ("Send my PC to sleep, pronto.", "sleep,PC"),
    ("I want my PC to sleep", "sleep,PC"),
    ("WINTER. I going away, please go to sleep.", "sleep,WINTER"),
    ("Have some sweet dreams, WINTER", "sleep,WINTER"),
    ("WINTER, I wanna listen to The King of Pop's Thriller song", "play,on_youtube"),
    ("I want to watch some PewDiePie videos. Play them for me please.", "play,on_youtube"),
    ("Play Michael Jackson's Smooth Criminal lyrics.", "play,on_youtube"),
    ("You know yesterday I was playing football and I fell. I fell so hard it got my knees hurt. They started to bleed and it was really painful. Dude only I know how I survived.", "default"),
    ("please play Leon on song for me.", "play,on_youtube"),
    ("please play Leon on song for me from my computer", "play,music"),
    ("winter show me some latest AgneTheGreat developments to his steam game", "play,on_youtube"),
    ("Sebastian Lague has uploaded a new video buddy", "play,on_youtube"),
    ("I made my own game engine", "search,youtube"),
    ("Buddy check if There's a new video of RGbucklist, it's been a long time", "play,on_youtube"),
    ("im going a vacation son. U know the deed. Take some rest", "sleep,PC")
]

import torch
s = sample()
s.load(torch.load("bin\\clis.bin"), True)

import pickle
from colorama import Style, Fore, init
init(autoreset=True)

with open("data\\clis\\classes.bin", "rb") as f:
    classes = pickle.load(f)

for text, expec_tag in test:
    # enctxt = enc.encode(text)
    # enctxt.extend([4279] * (64 - len(enctxt)))
    # tag, conf = s.predict(torch.tensor(enctxt, dtype=torch.float32), classes)
    # tag, conf = s.predict(torch.tensor(enctxt, dtype=torch.float32), classes)

    enctxt = one_hot_encoding(text, vocab) # encoded text
    tag, conf = s.predict(enctxt, classes)

    print(text)
    print(enctxt)
    if expec_tag == tag:
        print(tag, conf)

    else:
        print(f"{Fore.RED}{Style.BRIGHT}{tag, conf}")
        print(f"Correct tag: {Fore.GREEN}{Style.BRIGHT}{expec_tag}")
    print()
    print("-"*100, "\n")

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
    conversation_path = "bin\\cache\\claw.txt")

while True:
    try:
        i = input("> ")

        if i.strip() == "":
            continue

        elif i == "exit":
            break

        out = llm.generate(i)
        dprint(out)

    except Exception as e:
        print(e.with_traceback())
        break
"""
