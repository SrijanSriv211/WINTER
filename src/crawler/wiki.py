from urllib.parse import urlparse
from bs4 import BeautifulSoup
import wikipedia, requests, time, re

DEFAULT_OUTPUT = 'output.txt'
DEFAULT_INTERVAL = 5.0  # interval between requests (seconds)
DEFAULT_ARTICLES_LIMIT = 1  # total number articles to be extrated
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'

visited_urls = set()  # all urls already visited, to not visit twice
pending_urls = []  # queue

def load_urls(session_file):
    """Resume previous session if any, load visited URLs"""

    try:
        with open(session_file, "r", encoding="utf-8") as fin:
            for line in fin:
                visited_urls.add(line.strip())
    except FileNotFoundError:
        pass

def scrap(base_url, article, output_file, session_file):
    """Represents one request per article"""

    full_url = base_url + article
    try:
        r = requests.get(full_url, headers={'User-Agent': USER_AGENT})

    except requests.exceptions.ConnectionError:
        print("Check your Internet connection")
        input("Press [ENTER] to continue to the next request.")
        return

    if r.status_code not in (200, 404):
        print("Failed to request page (code {})".format(r.status_code))
        input("Press [ENTER] to continue to the next request.")
        return

    soup = BeautifulSoup(r.text, 'html.parser')
    content = soup.find('div', {'id':'mw-content-text'})

    with open(session_file, 'a', encoding="utf-8") as fout:
        fout.write(full_url + '\n')  # log URL to session file

    # add new related articles to queue
    # check if are actual articles URL
    for a in content.find_all('a'):
        href = a.get('href')
        if not href:
            continue

        if href[0:6] != '/wiki/':  # allow only article pages
            continue

        elif ':' in href:  # ignore special articles e.g. 'Special:'
            continue

        elif href[-4:] in ".png .jpg .jpeg .svg":  # ignore image files inside articles
            continue

        elif base_url + href in visited_urls:  # already visited
            continue

        if href in pending_urls:  # already added to queue
            continue

        pending_urls.append(href)

    # skip if already added text from this article, as continuing session
    if full_url in visited_urls:
        return

    visited_urls.add(full_url)

    try:
        title = article[5+1:].replace("_", " ")
        wiki = wikipedia.page(title)
        text = "".join(wiki.content.split("\t"))
        text = text.replace("==", "").replace("\n\n", "\n")[:-12]

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    except Exception:
        parenthesis_regex = re.compile('\(.+?\)')  # to remove parenthesis content
        citations_regex = re.compile('\[.+?\]')  # to remove citations, e.g. [1]

        # get plain text from each <p>
        p_list = content.find_all('p')
        with open(output_file, 'a', encoding='utf-8') as fout:
            for p in p_list:
                text = p.get_text().strip()
                text = parenthesis_regex.sub('', text)
                text = citations_regex.sub('', text)
                if text:
                    text = " ".join(text.split()).replace(" , ", ", ").replace(",,", ",").replace(" . ", ". ")
                    fout.write(text + "\n")  # extra line between paragraphs
            fout.write("\n\n")  # extra line between paragraphs

def main(initial_url, articles_limit, interval, output_file):
    """ Main loop, single thread """

    minutes_estimate = interval * articles_limit / 60
    print("This session will take {:.1f} minute(s) to download {} article(s):".format(minutes_estimate, articles_limit))
    print("\t(Press CTRL+C to pause)\n")
    session_file = "session_" + output_file
    load_urls(session_file)  # load previous session (if any)
    base_url = '{uri.scheme}://{uri.netloc}'.format(uri=urlparse(initial_url))
    initial_url = initial_url[len(base_url):]
    pending_urls.append(initial_url)

    counter = 0
    while len(pending_urls) > 0:
        try:
            counter += 1
            if counter > articles_limit:
                break

            try:
                next_url = pending_urls.pop(0)

            except IndexError:
                break

            time.sleep(interval)
            article_format = next_url.replace('/wiki/', '')[:35]
            print("{:<7} {}".format(counter, article_format))
            scrap(base_url, next_url, output_file, session_file)

        except KeyboardInterrupt:
            input("\n> PAUSED. Press [ENTER] to continue...\n")
            counter -= 1

    print("Finished!")

n_articles = 1
interval = 1
output = "wiki.txt"

init = [
    "https://en.wikipedia.org/wiki/Shah_Rukh_Khan",
    "https://en.wikipedia.org/wiki/India",
    "https://en.wikipedia.org/wiki/Tom_Cruise",
    "https://en.wikipedia.org/wiki/Dan_Houser",
    "https://en.wikipedia.org/wiki/Sam_Houser",
    "https://en.wikipedia.org/wiki/Take-Two_Interactive",
    "https://en.wikipedia.org/wiki/New_York_City",
    "https://en.wikipedia.org/wiki/Grand_Theft_Auto",
    "https://en.wikipedia.org/wiki/Red_Dead",
    "https://en.wikipedia.org/wiki/Midnight_Club",
    "https://en.wikipedia.org/wiki/Rockstar_Games",
    "https://en.wikipedia.org/wiki/Tony_Stark_(Marvel_Cinematic_Universe)",
    "https://en.wikipedia.org/wiki/Robert_Downey_Jr.",
    "https://en.wikipedia.org/wiki/Marvel_Cinematic_Universe",
    "https://en.wikipedia.org/wiki/Google",
    "https://en.wikipedia.org/wiki/YouTube",
    "https://en.wikipedia.org/wiki/Elon_Musk",
    "https://en.wikipedia.org/wiki/Isaac_Newton",
    "https://en.wikipedia.org/wiki/Apple_Inc.",
    "https://en.wikipedia.org/wiki/Steve_Jobs",
    "https://en.wikipedia.org/wiki/Tesla,_Inc.",
    "https://en.wikipedia.org/wiki/Facebook",
    "https://en.wikipedia.org/wiki/Mark_Zuckerberg",
    "https://en.wikipedia.org/wiki/Harvard_College",
    "https://en.wikipedia.org/wiki/OpenAI",
    "https://en.wikipedia.org/wiki/DALL-E",
    "https://en.wikipedia.org/wiki/Sora_(text-to-video_model)",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Sam_Altman",
    "https://en.wikipedia.org/wiki/Bill_Gates"
    "https://en.wikipedia.org/wiki/Warren_Buffett",
    "https://en.wikipedia.org/wiki/Michael_Jackson"
]

for i in init:
    main(i, n_articles, interval, output)
    pending_urls = []
