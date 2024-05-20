from src.WINTER.features.features import Features
from src.vendor.GATw import alphabet

model = alphabet.classification.Sample("bin\\skills.pth")
feature = Features("data\\skills.json")

model.load()
feature.load()

test = [
    "please make the volume one hundred percent.",
    "turn the volume to zero percent.",
    "I think that things won't be any perfect than a quite PC",
    "I think that things won't be any perfect than a quite PC. Mute this device please.",
    "I don't think that things won't be any perfect than a quite PC.",
    "set volume one hundred percent.",
    "set the volume to zero percent.",
    "mute my pc for a minute",
    "unmute my PC now.",
    "increase the volume of this PC.",
    "please increase the volume",
    "please increase the volume by ten percent.",
    "please reduce the volume by fifteen points.",
    "set volume fifty nine",
    "pronto, I want you to set the volume slider to sixty nine.",
    "aalkjsdflkja9ierfasdfjlkaj adjfkkladjsfkl",
    "You know WINTER, my system has two button. One for sleep and the other one is for lock. No matter what you choose, they will perform the same function. Try it if you want.",
    "Lock this PC",
    "ya would you please put my system on sleep I'm going away for some time actually.",
    "please put my PC a lock. I won't be here.",
    "I think you should now send this PC to a quite deep sleep.",
    "Go to sleep WINTER.",
    "Sleep WINTER",
    "Send my PC to sleep, pronto.",
    "I want my PC to sleep",
    "WINTER. I going away, please go to sleep.",
    "Have some sweet dreams, WINTER",
    "WINTER, I wanna listen to The King of Pop's Thriller song",
    "I want to watch some PewDiePie videos. Play them for me please.",
    "Play Michael Jackson's Smooth Criminal lyrics.",
    "You know yesterday I was playing football and I fell. I fell so hard it got my knees hurt. They started to bleed and it was really painful. Dude only I know how I survived.",
    "please play Leon on song for me.",
    "please play Leon on song for me from my computer",
]

for i in test:
    print(i)

    feature.execute(i, model.predict(i), False)
    print("-"*100)
