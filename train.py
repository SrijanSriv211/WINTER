from src.models.rnn import train, model as rnn
from src.shared import nltk_utils
import torch, json

#############################################
################# Train RNN #################
#############################################

with open("data\\skills.json", "r", encoding="utf-8") as f:
    obj = json.load(f)

classes = []
vocab = []
xy = [] # x = pattern, y = tag

for intent in obj["skills"]:
    classes.append(intent["skill"])

    for pattern in intent["patterns"]:
        tokenized_words = nltk_utils.tokenize(pattern)
        vocab.extend(tokenized_words)
        xy.append((pattern, intent["skill"]))

# lemmatize, lower each word and remove unnecessary chars.
vocab = nltk_utils.remove_special_chars(vocab)

# remove duplicates and sort
vocab = sorted(set(vocab))
classes = sorted(set(classes))

# create dataset for training
data = [(nltk_utils.one_hot_encoding(x, vocab), classes.index(y)) for x, y in xy]

# configure model
rnn.RNNConfig.input_size = len(vocab)
rnn.RNNConfig.n_hidden = 16
rnn.RNNConfig.n_layer = 4
rnn.RNNConfig.output_size = len(classes)
rnn.RNNConfig.device = "cpu"

# train model
t = train.train(64)
t.prepare(data)
out = t.train(0.001, 2000)
out["classes"] = classes
out["vocab"] = vocab

"""
Acronyms for training:
1. 'is' stands for 'intended skill'. For eg, 'is9' = 'intended skill 9' meaning the model has 9 skills.
2. 'lo' stands for 'logical operator'. For eg, 'lo_and' = 'logical operator and'.
3. 'cl' stands for 'core language'. For eg, 'cl100k' = 'core language 100k' meaning it is a tokenizer with 100k tokens.
"""
torch.save(out, "bin\\models\\is9.pth")
