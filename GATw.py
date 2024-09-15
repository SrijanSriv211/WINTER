from src.models.gpt import train, model as gpt
from src.tokenizers.bpe.regex import RegexTokenizer
from src.shared.utils import dprint
import torch, json, os

# raw_data = []

# x = os.listdir("data")
# x.remove("clis.json")
# x.remove("lo.json")

# for i in x:
#     with open(f"data\\{i}", "r", encoding="utf-8") as f:
#         raw_data.append(f.read())

# with open("data\\clis.json", "r", encoding="utf-8") as f:
#     obj = json.load(f)

# for intent in obj["clis"]:
#     if intent["skill"] == "default" or intent["patterns"] == [""] or intent["patterns"] == []:
#         continue

#     raw_data.extend(intent["patterns"])

tokenizer = RegexTokenizer()
tokenizer.load("bin\\models\\cl2k.model")
# tokenizer.train("\n".join(raw_data), 2279, verbose=True)
tokenizer.register_special_tokens({"<|startoftext|>": 2279, "<|endoftext|>": 2280})
# tokenizer.save("bin\\models\\cl2k")

gpt.GPTConfig.vocab_size = 2281 # tokenizer vocab size + number of special tokens
gpt.GPTConfig.block_size = 50
gpt.GPTConfig.n_layer = 2
gpt.GPTConfig.n_head = 2
gpt.GPTConfig.n_embd = 8
gpt.GPTConfig.device = "cpu"

# out = torch.load("bin\\models\\claw.pth")
# gpt.GPTConfig.device = out["device"]
# gpt.GPTConfig.n_embd = out["config"]["n_embd"]
# gpt.GPTConfig.n_head = out["config"]["n_head"]
# gpt.GPTConfig.n_layer = out["config"]["n_layer"]
# gpt.GPTConfig.block_size = out["config"]["block_size"]
# gpt.GPTConfig.dropout = out["config"]["dropout"]
# gpt.GPTConfig.vocab_size = out["config"]["vocab_size"]

with open("data\\claw.txt", "r", encoding="utf-8") as f:
    data = tokenizer.encode(f.read(), allowed_special="all")

t = train.train(32)
# t.from_pretrained(out["state_dict"])
t.prepare(data, 1)
out = t.train(4e-3, 300_000, 1000, checkpoints = {"path": "bin\\models\\checkpoints", "name": "claw", "interval": 2000})
t.save(out, "bin\\models\\claw")

i = gpt.inference(out, True)
dprint(tokenizer.decode(i.generate(tokenizer.encode("Hello WINTER!\n<|startoftext|>", allowed_special="all"), length=20)))

for _ in range(10):
    dprint(tokenizer.decode(i.generate(length=20)))
