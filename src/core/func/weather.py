import requests

# https://medium.com/analytics-vidhya/forecast-weather-using-python-e6f5519dc3c1
def get_weather(city="muzaffarpur"):
    return requests.get(f"https://wttr.in/{city}?format=%C")

# https://medium.com/analytics-vidhya/forecast-weather-using-python-e6f5519dc3c1
def get_temp(city="muzaffarpur"):
    res = requests.get(f"https://wttr.in/{city}?format=%t")
    return res.text[1:] if res.text[0] == "+" else res.text
