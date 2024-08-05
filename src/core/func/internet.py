import webbrowser, wikipedia

def search_on_google(query: str):
    webbrowser.open(f"https://www.google.com/search?q={query.replace(" ", "+")}")

def search_on_wikipedia(topic: str):
    search_on_google(f"{topic} wikipedia")
    webbrowser.open(f"https://en.wikipedia.org/wiki/{topic.replace(" ", "")}")
    return wikipedia.summary(topic, sentences=3)
