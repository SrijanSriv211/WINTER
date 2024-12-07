import sys
sys.path.insert(0, "D:\\Dev Projects\\WINTER")

from src.models.encoder import Encoder
from src.shared.utils import prepare_data
import json, os

enc = Encoder()
enc.load("bin\\cl8k.bin")

data = []
files = os.listdir("data\\claw\\raw")
files.remove("gpt-2.txt")

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
	with open(f"data\\claw\\raw\\{i}", "r", encoding="utf-8") as f:
		data.append(f.read())

data = "\n\n".join(data)
print(f"{(len(data)/1e6)}M total chars", f"{(len(set(data)))} unique chars")

prepare_data(enc.encode(data, allowed_special="all"), "data\\claw", 0.9, distribution=None)
