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

# # Perform one-hot-encoding on numbers/vectors
# def one_hot_encoding(x, vocab, sent=False):
#     """
#     `x`: (`int` or `str`) -> Either a number or tokenizer sentences
#     'vocab': (list[`int` or `str`]) -> List of numbers or list of all words
#     `sent`: bool -> If true then treat `x` and `vocab` as `str` and `list[str]` respecticely, otherwise treat them as `int` and `list[int]` respectively
#     """

#     if sent:
#         x = set([lemmatize(word) for word in x])

#     encoding = numpy.zeros(len(vocab), dtype=numpy.float32)

#     for idx, w in enumerate(vocab):
#         if w in x:
#             encoding[idx] = 1

#     return encoding

def one_hot_encoding(x, vocab):
    encoding = numpy.zeros(len(vocab), dtype=numpy.float32)

    for idx, w in enumerate(vocab):
        if w in x:
            encoding[idx] = 1

    return encoding
