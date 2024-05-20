from nltk.stem import WordNetLemmatizer
import numpy, nltk

Lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence.strip())

def lemmatize(word):
    return Lemmatizer.lemmatize(word.lower().strip())

def remove_special_chars(tokens):
    #NOTE: Don't remove [+, -, *, /], because they are math symbols.
    ignore_chars = '''!{};:'"\,<>?@#$&_~'''
    return [word for word in tokens if word not in ignore_chars]

def one_hot_encoding(tokenized_sentence, words):
    sentence_words = set([lemmatize(word) for word in tokenized_sentence])
    encoding = numpy.zeros(len(words), dtype=numpy.float32)
    
    for idx, w in enumerate(words):
        if w in sentence_words:
            encoding[idx] = 1

    return encoding

# encoder: take a string, output a list of integers
def encode(text, stoi):
    return [stoi[c] for c in text]

# decoder: take a list of integers, output a string
def decode(tensor, itos):
    return ''.join([itos[i] for i in tensor])
