from nltk.stem import WordNetLemmatizer
import numpy, nltk

# nltk.download('punkt')

Lemmatizer = WordNetLemmatizer()

def tokenize(sentence: str):
    return nltk.word_tokenize(sentence.strip())

def lemmatize(word: str):
    return Lemmatizer.lemmatize(word.lower().strip())

def remove_special_chars(tokens):
    #NOTE: Don't remove [+, -, *, /], because they are math symbols.
    ignore_chars = '''!{};:'"\\,<>?@#$&_~'''
    return [word for word in tokens if word not in ignore_chars]

def one_hot_encoding(x, vocab):
    encoding = numpy.zeros(len(vocab), dtype=numpy.float32)

    for idx, w in enumerate(vocab):
        if w in x:
            encoding[idx] = 1

    return encoding
