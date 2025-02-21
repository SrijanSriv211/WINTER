import sys
sys.path.insert(0, "D:\\Dev Projects\\WINTER")

from src.encoder.word import Encoder
from src.shared.utils import prepare_data
import json, os

enc = Encoder()
enc.load("bin\\cl7k.bin")

"""
Fine tuning dataset
"""
data = []
files = os.listdir("data\\GATw\\raw")
[files.remove(i) for i in ["gpt-4.txt", "tweets.txt", "jokes.txt"]]

with open("data\\clis\\clis.json", "r", encoding="utf-8") as f:
	o = json.load(f)

patterns = []
for i in o:
	if i["patterns"] == []:
		continue

	patterns.extend(i["patterns"])
data.append("\n".join(patterns))
del patterns

for i in files:
	with open(f"data\\GATw\\raw\\{i}", "r", encoding="utf-8") as f:
		data.append(f.read())
data = "\n\n".join(data)

"""
Pretraining dataset
"""
# with open(f"data\\GATw\\raw\\gpt-4.txt", "r", encoding="utf-8") as f:
# 	data = f.read()

"""
Save dataset
"""

# with open("data\\tok_enc.txt", "w", encoding="utf-8") as f:
# 	f.write(data + "\n")

print(f"{(len(data)/1e6)}M total chars", f"{(len(set(data)))} unique chars")
prepare_data(enc.encode(data, allowed_special="all"), "data\\GATw", 1, distribution=None)
