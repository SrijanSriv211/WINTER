from src.models.gpt import train, model as gpt
from src.tokenizers.bpe.regex import RegexTokenizer
from src.shared.utils import dprint
import torch, json, os

# raw_data = []

# x = os.listdir("data")
# x.remove("clis.json")
# x.remove("lo.json")
# x.remove("data.txt")
# x.remove("LLM data.txt")
# x.remove("conversations.txt")

# with open("data\\clis.json", "r", encoding="utf-8") as f:
#     obj = json.load(f)

# for intent in obj["clis"]:
#     if intent["skill"] == "default" or intent["patterns"] == [""] or intent["patterns"] == []:
#         continue

#     raw_data.extend(intent["patterns"])

# for i in x:
#     with open(f"data\\{i}", "r", encoding="utf-8") as f:
#         raw_data.append(f.read())

# with open(f"data\\data.txt", "r", encoding="utf-8") as f:
#     t = f.read()
#     i = int(len(t) * 0.02) + 49000 - 250 - 171
#     raw_data.append(t[:i])

# with open(f"data\\LLM data.txt", "r", encoding="utf-8") as f:
#     t = f.read()
#     i = int(len(t) * 0.1) + 400000 + 170
#     raw_data.append(t[:i])

# with open(f"data\\conversations.txt", "r", encoding="utf-8") as f:
#     t = f.read()
#     i = int(len(t) * 0.01) - 50000 + 114
#     raw_data.append(t[:i])

tokenizer = RegexTokenizer()
tokenizer.load("bin\\models\\cl2k.model")
# tokenizer.train("\n".join(raw_data), 2279, verbose=True)
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
# t = train.train(32, out)
t = train.train(32)
t.prepare(data, 1)
out = t.train(4e-3, 1000)
t.save(out, "bin\\models\\claw")

i = gpt.inference(out, True)
dprint(tokenizer.decode(i.generate()))
