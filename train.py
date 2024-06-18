from src.vendor.GATw import RegexTokenizer, one_hot_encoding, rnn
import torch, json

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

# Creating vocabulary
tokenizer = RegexTokenizer()
# tokenizer.train('\n'.join(patterns), vocab_size=2048)
# tokenizer.save("bin\\models\\tok2k") # writes tok2k.model and tok2k.vocab
tokenizer.load("bin\\models\\tok2k.model") # loads the model back from disk

# Create dataset for training
data = [(one_hot_encoding(tokenizer.encode(x), list(tokenizer.vocab.keys())), classes.index(y)) for x, y in xy]

rnn.RNNConfig.input_size = len(tokenizer.vocab)
rnn.RNNConfig.n_hidden = 8
rnn.RNNConfig.n_layer = 2
rnn.RNNConfig.output_size = len(classes)
rnn.RNNConfig.device = "cpu"

rnn.RNNTrainConfig.n_steps = 1200
rnn.RNNTrainConfig.eval_interval = 200
rnn.RNNTrainConfig.eval_iters = 400

r = rnn.train(batch_size=128)
r.prepare(data)
x = r.train(lr=1e-3)
x["classes"] = classes

torch.save(x, "bin\\models\\skills.pth")
