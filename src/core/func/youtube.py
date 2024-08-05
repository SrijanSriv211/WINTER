import webbrowser, requests

# Search and play a random most recent youtube video from subscribed channels.
def play_random_youtube_video():
    pass

# https://github.com/Ankit404butfound/PyWhatKit/blob/master/pywhatkit/misc.py
def play_youtube_video(topic):
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
