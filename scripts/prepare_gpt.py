import sys
sys.path.insert(0, "D:\\Dev Projects\\WINTER")

from src.models.encoder import Encoder
from src.shared.utils import prepare_data
import json

enc = Encoder()
enc.load("bin\\cl8k.bin")

def get_foundational_dataset():
	data = []

	with open("data\\clis\\clis.json", "r", encoding="utf-8") as f:
		obj = json.load(f)

	for intent in obj:
		if intent["patterns"] == []:
			continue

		data.extend(intent["patterns"])

	with open("data\\claw\\raw\\Knowledge copy.txt", "r", encoding="utf-8") as f:
		data.append(f.read())

	with open("data\\claw\\raw\\General notes.txt", "r", encoding="utf-8") as f:
		data.append(f.read())

	with open("data\\claw\\raw\\facts.txt", "r", encoding="utf-8") as f:
		data.append(f.read())

	with open("data\\claw\\raw\\LLM data.txt", "r", encoding="utf-8") as f:
		data.append(f.read())

	with open("data\\claw\\raw\\jokes.txt", "r", encoding="utf-8") as f:
		data.append(f.read())

	with open("data\\claw\\raw\\data.txt", "r", encoding="utf-8") as f:
		data.append(f.read())

	data = "\n\n".join(data)
	print(f"{(len(data)/1e6)}M total chars", f"{(len(set(data)))} unique chars")
	return data

def get_conversational_dataset():
	with open("data\\claw\\raw\\claw.txt", "r", encoding="utf-8") as f:
		return f.read()

prepare_data(enc.encode(get_foundational_dataset(), allowed_special="all"), "data\\claw", 0.95)
