from src.tokenizers.bpe.regex import RegexTokenizer
from src.models.gpt import train as train_gpt, model as gpt
from src.models.rnn import train as train_rnn, model as rnn
from src.shared.utils import dprint
from src.shared import nltk_utils
import torch, json, os

####################################################
#################### Train Claw ####################
####################################################

def train_claw():
    tokenizer = RegexTokenizer()
    tokenizer.load("bin\\models\\cl2k.model")
    # tokenizer.train(data, 2279, verbose=True)
    tokenizer.register_special_tokens({"<|clis|>": 2279, "<|startoftext|>": 2280, "<|endoftext|>": 2281})
    # tokenizer.save("bin\\models\\cl2k")

    gpt.GPTConfig.vocab_size = 2482 # tokenizer vocab size + number of special tokens
    gpt.GPTConfig.block_size = 50
    gpt.GPTConfig.n_layer = 2
    gpt.GPTConfig.n_head = 2
    gpt.GPTConfig.n_embd = 32
    gpt.GPTConfig.device = "cpu"

    with open("data\\claw.txt", "r", encoding="utf-8") as f:
        data = tokenizer.encode(f.read(), allowed_special="all")

    # out = torch.load("bin\\models\\claw.pth")
    # t = train_gpt.train(32, out)
    t = train_gpt.train(32)
    t.prepare(data, 1)
    out = t.train(4e-3, 1000)
    t.save(out, "bin\\models\\claw")

    i = gpt.inference(out, True)
    dprint(tokenizer.decode(i.generate()))

####################################################
#################### Train Clis ####################
####################################################

def train_clis():
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
    rnn.RNNConfig.n_layer = 2
    rnn.RNNConfig.output_size = len(classes)
    rnn.RNNConfig.device = "cpu"

    # train model
    t = train_rnn.train(64)
    t.prepare(data)
    out = t.train(1e-3, 5000)
    out["classes"] = classes
    out["vocab"] = vocab

    # 'clis1k10' means 'core language intentions 1000 tokens with 10 classes'
    # 'cl1kis10' = 'clis1k10'. Since the name is very simple number of tokens and the prefix name are interchangable.
    torch.save(out, f"bin\\models\\clis1k{len(classes)}.pth")
