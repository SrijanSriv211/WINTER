from src.vendor.GATw import RegexTokenizer, rnn
# from src.vendor.GATw import gpt
import torch, json, os

with open("data\\skills.json", "r", encoding="utf-8") as f:
    obj = json.load(f)

patterns = []
classes = []
xy = []

for intent in obj["skills"]:
    if intent["skill"] == "default":
        continue

    classes.append(intent["skill"])
    patterns.extend(intent["patterns"])

    for pattern in intent["patterns"]:
        xy.append((pattern, intent["skill"]))

print("Creating vocabulary")
tokenizer = RegexTokenizer()
# tokenizer.train('\n'.join(patterns), vocab_size=2048)
# tokenizer.save("bin\\models\\2k") # writes 2k.model and 2k.vocab
tokenizer.load("bin\\models\\2k.model") # loads the model back from disk

def one_hot_encoding(x, vocab):
    import numpy

    encoding = numpy.zeros(len(vocab), dtype=numpy.float32)
    
    for idx, w in enumerate(vocab):
        if w in x:
            encoding[idx] = 1

    return encoding

# Create dataset for training
data = [(one_hot_encoding(tokenizer.encode(x), list(tokenizer.vocab.keys())), classes.index(y)) for x, y in xy]

rnn.RNNConfig.input_size = 2048
rnn.RNNConfig.n_hidden = 8
rnn.RNNConfig.n_layer = 2
rnn.RNNConfig.output_size = len(classes)
rnn.RNNConfig.device = "cpu"

r = rnn.train(batch_size=64)
r.prepare(data)
x = r.train(lr=1e-3)

torch.save(x, "bin\\models\\skills.pth")


# gpt.GPTConfig.vocab_size = len(data)
# gpt.GPTConfig.n_layer = 8
# gpt.GPTConfig.n_embd = 8
# gpt.GPTConfig.n_head = 8
# gpt.GPTConfig.dropout = 0
# gpt.GPTConfig.block_size = 500

# w = gpt.train(batch_size=32)
# w.prepare(data, 0.8)
# x = w.train(1e-3)



# from src.vendor.GATw import alphabet
# from src.vendor.GATw import write

# # ================================== Train alphabet ==================================

# # at = alphabet.classification.Train(
# #     n_layer = 1,
# #     n_hidden = 8,
# #     lr = 4e-3,
# #     batch_size = 64,
# #     model = "RNN"
# # )
# # at.preprocess("data\\skills.json", metadata=("skills", "skill", "patterns"), data_division=None)
# # at.train(
# #     n_steps = 4000,
# #     eval_interval = 1000,
# #     eval_iters = 800,
# #     n_loss_digits = 7
# # )
# # at.save("bin\\skills.pth")

# # ==================================== Train GATw ====================================

# wt = write.Train(
#     n_layer = 8,
#     n_embd = 8,
#     n_head = 8,
#     lr = 1e-3,
#     dropout = 0,
#     block_size = 500,
#     batch_size = 32
# )

# wt.preprocess("data\\data.txt")
# wt.train(
#     n_steps = int(3e5),
#     eval_interval = int(1e4),
#     eval_iters = int(1e4),
#     checkpoint_path = "bin\\models\\GATw\\checkpoints\\checkpoints.pth",
#     checkpoint_interval = int(1e4)
# )
# wt.save("bin\\models\\GATw\\GATw.pth")
