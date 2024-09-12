import time, os

# dprint -> delay print
# ChatGPT like print effect.
def dprint(text, delay=0.001):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print("\n")

def get_bin_path():
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Navigate up the directory tree four times
    parent_dir = current_file_path
    rel_current_path = os.path.relpath(__file__)
    for _ in range(len(rel_current_path.split(os.sep))):
        parent_dir = os.path.dirname(parent_dir)

    # Append "cache" directory to the parent directory
    return os.path.join(parent_dir, "bin")

def calc_total_time(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    hours, minutes, seconds = int(hour), int(min), int(sec)

    t = [
        f"{hours} hour" + ("s" if hours > 1 else "") if hours > 0 else None,
        f"{minutes} minute" + ("s" if minutes > 1 else "") if minutes > 0 else None,
        f"{seconds} second" + ("s" if seconds > 1 else "") if seconds > 0 else None
    ]
    t = list(filter(None, t))

    print("Total time:", ", ".join(t) if t else "0 seconds")
