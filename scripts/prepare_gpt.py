import sys
sys.path.insert(0, "D:\\Dev Projects\\WINTER")

from src.models.encoder import Encoder
from src.shared.utils import prepare_data
import os

enc = Encoder()
enc.load("bin\\cl8k.bin")

def get_foundational_dataset():
	data = []

	files = os.listdir("data\\claw\\raw")
	files.remove("claw.txt")

	for i in files:
		with open(f"data\\claw\\raw\\{i}", "r", encoding="utf-8") as f:
			data.append(f.read()[:200_000_000] if i == "gpt-2.txt" else f.read())

	data = "\n\n".join(data)
	print(f"{(len(data)/1e6)}M total chars", f"{(len(set(data)))} unique chars")
	return data

def get_conversational_dataset():
	with open("data\\claw\\raw\\claw.txt", "r", encoding="utf-8") as f:
		return f.read()

prepare_data(enc.encode(get_foundational_dataset(), allowed_special="all"), "data\\claw", 0.9, distribution=None)
