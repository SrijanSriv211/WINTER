import ctypes, os, screen_brightness_control as sbc

def change_wallpaper(path):
    abspath = os.path.abspath(path)
    ctypes.windll.user32.SystemParametersInfoW(20, 0, abspath , 0)
    return abspath

def get_brightness():
    return sbc.get_brightness()[0]

def set_brightness(level):
    if (100 >= level > 0) == False:
        level = 100

    sbc.set_brightness(level)

def change_brightness(magnitude):
    set_brightness(get_brightness() + magnitude)
