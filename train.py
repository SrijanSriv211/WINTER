from src.models.rnn import train, model as rnn
from src.shared import nltk_utils
import torch, json

####################################################
################# Train skills RNN #################
####################################################

with open("data\\clis.json", "r", encoding="utf-8") as f:
    obj = json.load(f)

classes = []
vocab = []
xy = [] # x = pattern, y = tag

for intent in obj["clis"]:
    skill = intent["skill"]
    patterns = intent["patterns"]

    if skill != "default" and patterns == [""]:
        continue

    classes.append(skill)

    for pattern in patterns:
        tokenized_words = nltk_utils.tokenize(pattern)
        vocab.extend(tokenized_words)
        xy.append((pattern, skill))

# lemmatize, lower each word and remove unnecessary chars.
vocab = nltk_utils.remove_special_chars(vocab)

# remove duplicates and sort
vocab = sorted(set(vocab))
classes = sorted(set(classes))

# create dataset for training
data = [(nltk_utils.one_hot_encoding(x, vocab), classes.index(y)) for x, y in xy]

# configure model
rnn.RNNConfig.input_size = len(vocab)
rnn.RNNConfig.n_hidden = 8
rnn.RNNConfig.n_layer = 1
rnn.RNNConfig.output_size = len(classes)
rnn.RNNConfig.device = "cpu"

# train model
t = train.train(64)
t.prepare(data)
out = t.train(1e-3, 2000)
out["classes"] = classes
out["vocab"] = vocab

# 'clis1k10' means 'core language intentions 1000 tokens with 10 classes'
# 'cl1kis10' = 'clis1k10'. Since the name is very simple number of tokens and the prefix name are interchangable.
torch.save(out, f"bin\\models\\clis1k{len(classes)}.pth")
