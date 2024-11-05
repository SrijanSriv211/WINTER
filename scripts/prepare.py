import sys
sys.path.insert(0, "D:\\Dev Projects\\WINTER")

from src.tokenizers.bpe import RegexTokenizer
from src.models.rnn import sample

enc = RegexTokenizer()
enc.load("bin\\cl4k.model")

from src.shared.utils import prepare_data
from src.shared.nltk_utils import tokenize, remove_special_chars, one_hot_encoding
import json

with open("data\\clis\\clis.json", "r", encoding="utf-8") as f:
    obj = json.load(f)

classes = []
xy = [] # x = pattern, y = tag

for intent in obj["clis"]:
	skill = intent["skill"]
	patterns = intent["patterns"]

	if patterns == [""]:
		continue
    
	classes.append(skill)

	for pattern in patterns:
		tokenized_pattern = enc.encode(pattern, allowed_special="all")
		tokenized_pattern.extend([4279] * (64 - len(tokenized_pattern)))
		xy.append((tokenized_pattern, skill))

# print(len(max([x for x, y in xy], key=len)))
print(len(classes))
print(len(xy))

import torch
data = [(torch.tensor(x, dtype=torch.long), torch.tensor(classes.index(y), dtype=torch.long)) for x, y in xy]

# classes = []
# vocab = []
# xy = [] # x = pattern, y = tag

# for intent in obj["clis"]:
#     skill = intent["skill"]
#     patterns = intent["patterns"]

#     if skill != "default" and patterns == [""]:
#         continue

#     classes.append(skill)

#     for pattern in patterns:
#         tokenized_words = tokenize(pattern)
#         vocab.extend(tokenized_words)
#         xy.append((pattern, skill))

# # lemmatize, lower each word and remove unnecessary chars.
# vocab = remove_special_chars(vocab)

# # remove duplicates and sort
# vocab = sorted(set(vocab))
# classes = sorted(set(classes))

# import torch
# # create dataset for training
# data = [(torch.tensor(one_hot_encoding(x, vocab), dtype=torch.long), torch.tensor(classes.index(y), dtype=torch.long)) for x, y in xy]

prepare_data(data, "data\\clis", 1, False)

import pickle
with open("data\\clis\\classes.bin", "wb") as f:
    pickle.dump(classes, f)
