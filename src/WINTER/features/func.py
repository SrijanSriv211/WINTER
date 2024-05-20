import webbrowser, wikipedia, requests
from googletrans import Translator

# init modules
translator = Translator()

def play_video(title):
    pass

def play_music(title):
    pass

def playgames(name):
    pass

# https://github.com/Ankit404butfound/PyWhatKit/blob/master/pywhatkit/misc.py
def youtube(topic):
    url = f"https://www.youtube.com/results?q={topic}"
    count = 0
    cont = requests.get(url, timeout=5)
    data = cont.content
    data = str(data)
    lst = data.split('"')

    for i in lst:
        count += 1
        if i == "WEB_PAGE_TYPE_WATCH":
            break

    if lst[count - 5] == "/results":
        raise Exception("No Video Found for this Topic!")

    webbrowser.open(f"https://www.youtube.com{lst[count - 5]}")
    return f"https://www.youtube.com{lst[count - 5]}"

def search_on_wikipedia(topic):
    return wikipedia.summary(topic, sentences=3)

def translate(sentence, dest="en"):
    out = translator.translate(sentence, dest=dest)
    return out.text
