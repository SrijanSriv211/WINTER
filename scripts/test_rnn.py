import sys
sys.path.insert(0, "D:\\Dev Projects\\WINTER")

from src.shared.nltk_utils import one_hot_encoding
from src.models.encoder import Encoder
from colorama import Style, Fore, init
from src.models.rnn import sample
import warnings, pickle, torch

# supress pytorch's future warning:
# You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly.
# It is possible to construct malicious pickle data which will execute arbitrary code during unpickling
# (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details).
# In a future release, the default value for `weights_only` will be flipped to `True`.
# This limits the functions that could be executed during unpickling.
# Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`.
# We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file.
# Please open an issue on GitHub for any issues related to this experimental feature.
warnings.filterwarnings("ignore", category=FutureWarning)

init(autoreset=True)

# test = [
#     ("WINTER play me some BBS videos.", "play,on_youtube"),
#     ("open Google chrome and Can you please search google for me?", "search,google"),
#     ("What is a game engine?", "search,google"),
#     ("please search on google What is a game engine?", "search,google"),
#     ("Google Chrome", "open,app"),
#     ("Please start Unity game engine for me please", "open,app"),
#     ("open youtube.com please", "open,website"),
#     ("if you don't mind would you please search google about shahrukh khan", "search,google"),
#     ("how to start a youtube channel", "search,google"),
#     ("what is the capital of paris", "search,google"),
#     ("please search on google what is the capital of paris", "search,google"),
#     ("reload this PC, it needs some freshness", "restart,PC"),
#     ("i'm gonna be away for a while, please lock this pc", "sleep,PC"),
#     ("shutdown my workstation please", "shutdown,PC"),
#     ("shutdown google chrome", "shutdown,app"),
#     ("kill this app", "shutdown,app"),
#     ("tell me current date please", "date,current"),
#     ("tell me today's time", "time,current"),
#     ("is it monday today?", "day,current"),
#     ("Please start Unity game engine for me please", "open,app"),
#     ("start chrome.exe", "open,app"),
#     ("How do you make a game engine and remember to make it in c++", "search,google"),
#     ("open chrome and search on the internet How do you make a game engine", "search,google"),
#     ("please make the volume one hundred percent.", "volume,specific_level"),
#     ("turn the volume to zero percent.", "volume,mute"),
#     ("I think that things won't be any perfect than a quite PC", "volume,mute"),
#     ("I think that things won't be any perfect than a quite PC. Mute this device please.", "volume,mute"),
#     ("I don't think that things won't be any perfect than a quite PC.", "volume,unmute"),
#     ("set volume one hundred percent.", "volume,specific_level"),
#     ("set the volume to zero percent.", "volume,mute"),
#     ("mute my pc for a minute", "volume,mute"),
#     ("unmute my PC now.", "volume,unmute"),
#     ("increase the volume of this PC.", "volume,up"),
#     ("please increase the volume", "volume,up"),
#     ("please increase the volume by ten percent.", "volume,up"),
#     ("please reduce the volume by fifteen points.", "volume,down"),
#     ("set volume fifty nine", "volume,specific_level"),
#     ("pronto, I want you to set the volume slider to sixty nine.", "volume,specific_level"),
#     ("aalkjsdflkja9ierfasdfjlkaj adjfkkladjsfkl", "default"),
#     ("You know WINTER, my system has two button. One for sleep and the other one is for lock. No matter what you choose, they will perform the same function. Try it if you want.", "sleep,PC"),
#     ("Lock this PC", "sleep,PC"),
#     ("ya would you please put my system on sleep I'm going away for some time actually.", "sleep,PC"),
#     ("please put my PC a lock. I won't be here.", "sleep,PC"),
#     ("I think you should now send this PC to a quite deep sleep.", "sleep,PC"),
#     ("Go to sleep WINTER.", "sleep,WINTER"),
#     ("Sleep WINTER", "sleep,WINTER"),
#     ("Send my PC to sleep, pronto.", "sleep,PC"),
#     ("I want my PC to sleep", "sleep,PC"),
#     ("WINTER. I going away, please go to sleep.", "sleep,WINTER"),
#     ("Have some sweet dreams, WINTER", "sleep,WINTER"),
#     ("WINTER, I wanna listen to The King of Pop's Thriller song", "play,on_youtube"),
#     ("I want to watch some PewDiePie videos. Play them for me please.", "play,on_youtube"),
#     ("Play Michael Jackson's Smooth Criminal lyrics.", "play,on_youtube"),
#     ("You know yesterday I was playing football and I fell. I fell so hard it got my knees hurt. They started to bleed and it was really painful. Dude only I know how I survived.", "default"),
#     ("please play Leon on song for me.", "play,on_youtube"),
#     ("please play Leon on song for me from my computer", "play,music"),
#     ("winter show me some latest AgneTheGreat developments to his steam game", "play,on_youtube"),
#     ("Sebastian Lague has uploaded a new video buddy", "play,on_youtube"),
#     ("I made my own game engine", "search,youtube"),
#     ("Buddy check if There's a new video of RGbucklist, it's been a long time", "play,on_youtube"),
#     ("im going a vacation son. U know the deed. Take some rest", "sleep,PC")
# ]

s = sample()
s.load(torch.load("bin\\clis.bin"), True)

enc = Encoder()
enc.load("bin\\cl8k.bin")

with open("data\\clis\\classes.bin", "rb") as f:
    classes = pickle.load(f)

# for text, expec_tag in test:
#     enctxt = enc.encode(text.lower())
#     enctxt = one_hot_encoding(enctxt, list(range(4282))) # encoded text
#     tag, conf = s.predict(enctxt, classes)

#     print(text)
#     print(enctxt)
#     if expec_tag == tag:
#         print(tag, conf)

#     else:
#         print(f"{Fore.RED}{Style.BRIGHT}{tag, conf}")
#         print(f"Correct tag: {Fore.GREEN}{Style.BRIGHT}{expec_tag}")
#     print()
#     print("-"*100, "\n")

enctxt = enc.encode(input("> ").lower())
enctxt = one_hot_encoding(enctxt, list(range(4282))) # encoded text
tag, conf = s.predict(enctxt, classes)
print(tag, conf)
