import time, os

# dprint -> delay print
# ChatGPT like print effect.
def dprint(text, delay=0.001):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()

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
