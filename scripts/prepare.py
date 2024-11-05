import sys
sys.path.insert(0, "D:\\Dev Projects\\WINTER")

from src.shared.nltk_utils import one_hot_encoding
from src.tokenizers.bpe import RegexTokenizer
from src.shared.utils import prepare_data
import json

enc = RegexTokenizer()
enc.load("bin\\cl4k.model")

with open("data\\clis\\clis.json", "r", encoding="utf-8") as f:
    obj = json.load(f)

classes = []
xy = [] # x = pattern, y = tag

for intent in obj:
	skill = intent["skill"]
	patterns = intent["patterns"]

	if patterns == []:
		continue
    
	classes.append(skill)

	for pattern in patterns:
		xy.append((enc.encode(pattern, allowed_special="all"), skill))

print(len(classes))
print(len(xy))

import torch
# create dataset for training
data = [(torch.tensor(one_hot_encoding(x, list(range(4282))), dtype=torch.long), torch.tensor(classes.index(y), dtype=torch.long)) for x, y in xy]

prepare_data(data, "data\\clis", 1, False)

import pickle
with open("data\\clis\\classes.bin", "wb") as f:
    pickle.dump(classes, f)
