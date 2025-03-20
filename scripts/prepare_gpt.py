import sys
sys.path.insert(0, "D:\\Dev Projects\\WINTER")

from src.encoder.bytepair import Encoder
from src.shared.utils import prepare_data
import json, os

enc = Encoder()
enc.load("bin\\cl1k.bin")

"""
Pretraining dataset
"""
# data = []
# path = "data\\GATw\\base"
# files = os.listdir(path)
# [files.remove(i) for i in ["gpt-3.txt"]]

"""
Finetuning dataset
"""
data = []
path = "data\\GATw\\fine"
notebooks = ["General notes.txt", "Knowledge copy.txt", "The brilliant mind.txt"]
files = os.listdir(path)
[files.remove(i) for i in notebooks]

with open("data\\clis\\clis.json", "r", encoding="utf-8") as f:
	o = json.load(f)

patterns = []
for i in o:
	if i["patterns"] == []:
		continue

	patterns.extend(i["patterns"])
data.append("\n".join(patterns))
del patterns

for i in notebooks:
	with open(f"{path}\\{i}", "r", encoding="utf-8") as f:
		data.append(f"========== {i} ==========\n{f.read()}")

for i in files:
	with open(f"{path}\\{i}", "r", encoding="utf-8") as f:
		data.append(f"========== {i} ==========\n{f.read()}")
data = "\n\n".join(data)

"""
Save dataset
"""

# with open("data\\tok_enc.txt", "w", encoding="utf-8") as f:
# 	f.write(data + "\n")

print(f"{(len(data)/1e6)}M total chars", f"{(len(set(data)))} unique chars")
prepare_data(enc.encode(data, allowed_special="all"), "data\\GATw", 0.8, distribution=None)
