class CharTokenizer:
    def __init__(self, text):
        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))
        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

    # encoder: take a string, output a list of integers
    def encode(self, text, stoi):
        return [stoi[c] for c in text]

    # decoder: take a list of integers, output a string
    def decode(self, tensor, itos):
        return ''.join([itos[i] for i in tensor])
