from flask import Flask, render_template, request
from colorama import Style, Fore, init

init(autoreset=True)

# https://onlinetools.com/ascii/convert-text-to-ascii-art#tool
print(
f"""{Fore.YELLOW}{Style.BRIGHT}
  _    _      _ _         _____ _            
 | |  | |    | | |       |_   _( )           
 | |__| | ___| | | ___     | | |/ _ __ ___   
 |  __  |/ _ \\ | |/ _ \\    | |   | '_ ` _ \\  
 | |  | |  __/ | | (_) |  _| |_  | | | | | | 
 |_|  |_|\\___|_|_|\\___/  |_____| |_| |_| |_| 
                                             
"""

f"""{Fore.CYAN}{Style.BRIGHT}
           _       _            _ 
          (_)     | |          | |
 __      ___ _ __ | |_ ___ _ __| |
 \\ \\ /\\ / / | '_ \\| __/ _ \\ '__| |
  \\ V  V /| | | | | ||  __/ |  |_|
   \\_/\\_/ |_|_| |_|\\__\\___|_|  (_)
                                  
"""
)

print(
    f"{Fore.BLACK}{Style.BRIGHT} -- Witty Intelligence with Natural Emotions and Rationality --\n"
    "As my name suggests I am kind, helpful, witty, intelligent, emotional, empathetic, rational, clever and a charming personal assitant.\n"
    "I'm here to help you with any task possible because I'm a machine designed to accomplish a task..\n"
)

app = Flask(__name__, template_folder="src\\WINTER\\UI\\templates", static_folder="src\\WINTER\\UI\\static")
@app.get("/")
def index_get():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(port=8000)
