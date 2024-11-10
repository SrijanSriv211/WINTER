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

def save_distributed_data(path, name, data, distribution):
    distributed_data = [] # (path, data)

    # distribute the data
    if distribution:
        if not os.path.isdir(f"{path}\\{name}"):
            os.mkdir(f"{path}\\{name}")

        count = 0
        for i in range(0, len(data), distribution):
            distributed_data.append((f"{path}\\{name}\\{count}.bin", data[i:i+distribution]))
            count += 1

    else:
        distributed_data.append((f"{path}\\{name}.bin", data))

    # save the data
    for p, d in distributed_data:
        with open(p, "wb") as f:
            pickle.dump(d, f)

def prepare_data(encoded_data, path="data", data_division=1, convert_to_tensor=True, distribution=None):
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
        save_distributed_data(path, "val", val_data, distribution)
    save_distributed_data(path, "train", train_data[:] if 0 < data_division < 1 else data[:], distribution)
