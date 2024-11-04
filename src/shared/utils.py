import torch, pickle, time, sys, os

# dprint -> delay print
# ChatGPT like print effect.
def dprint(text, delay=0.001):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
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
    # Separate the integer part (for hours, minutes, and seconds) from the fractional part (for milliseconds)
    sec_int, millis = divmod(seconds, 1)
    millis = int(millis * 1000)  # Convert the fractional part to milliseconds

    min, sec = divmod(int(sec_int), 60)
    hour, min = divmod(min, 60)
    hours, minutes, seconds = int(hour), int(min), int(sec)

    t = [
        f"{hours} hour" + ("s" if hours > 1 else "") if hours > 0 else None,
        f"{minutes} minute" + ("s" if minutes > 1 else "") if minutes > 0 else None,
        f"{seconds} second" + ("s" if seconds > 1 else "") if seconds > 0 else None,
        f"{millis} ms" if millis > 0 else None
    ]
    t = list(filter(None, t))

    return ", ".join(t) if t else "0 seconds"

def prepare_data(encoded_data, path="bin", data_division=1, convert_to_tensor=True):
    """
    Generate `train.bin` and `val.bin` from encoded data.
    1. `encoded_data`: The encoded training text data.

    For eg,
    ```python
    encode(text, stoi=stoi)
    ```

    2. `data_division`: The first `(data_division * 100)%` will be train, rest val.

        If `data_division = 1`, then only `train.bin` will be saved and no data splitting will be done.

    3. `path`: Path where `train.bin` and `val.bin` will be saved
    """

    data = torch.tensor(encoded_data, dtype=torch.long) if convert_to_tensor else encoded_data

    # print the number of tokens
    print(f"{(len(data)/1e6)}M", "total tokens")

    if 0 < data_division < 1:
        # train and test splits
        n = int(data_division * len(data)) # the first (data_division * 100)% will be train, rest val
        train_data = data[:n]
        val_data = data[n:] if 0 < data_division < 1 else data[:n]

        print(f"{(len(train_data)/1e6)}M", "train tokens,", f"{(len(val_data)/1e6)}M", "test tokens")

        # save the data
        with open(f"{path}\\train.bin", "wb") as f:
            pickle.dump(train_data, f)

        with open(f"{path}\\val.bin", "wb") as f:
            pickle.dump(val_data, f)

    else:
        with open(f"{path}\\train.bin", "wb") as f:
            pickle.dump(data, f)
