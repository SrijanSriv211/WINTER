from src.models.gpt import train, model as gpt
from src.tokenizers.bpe.regex import RegexTokenizer
from src.shared.utils import dprint
import torch, json, os

raw_data = []

x = os.listdir("data")
x.remove("clis.json")
x.remove("lo.json")

with open("data\\clis.json", "r", encoding="utf-8") as f:
    obj = json.load(f)

for intent in obj["clis"]:
    if intent["skill"] == "default" or intent["patterns"] == [""] or intent["patterns"] == []:
        continue

    raw_data.extend(intent["patterns"])

for i in x:
    with open(f"data\\{i}", "r", encoding="utf-8") as f:
        raw_data.append(f.read())

data = "\n".join(raw_data)
sliced_upto = int(len(data)*0.1)
data = data[:sliced_upto]

tokenizer = RegexTokenizer()
# tokenizer.load("bin\\models\\cl2k.model")
tokenizer.train(data, 2279, verbose=True)
tokenizer.register_special_tokens({"<|startoftext|>": 2279, "<|endoftext|>": 2280})
tokenizer.save("bin\\models\\cl2k")

# gpt.GPTConfig.vocab_size = 2281 # tokenizer vocab size + number of special tokens
# gpt.GPTConfig.block_size = 1024
# gpt.GPTConfig.n_layer = 5
# gpt.GPTConfig.n_head = 5
# gpt.GPTConfig.n_embd = 128
# gpt.GPTConfig.device = "cpu"

# out = torch.load("bin\\models\\claw.pth")
# gpt.GPTConfig.device = out["device"]
# gpt.GPTConfig.n_embd = out["config"]["n_embd"]
# gpt.GPTConfig.n_head = out["config"]["n_head"]
# gpt.GPTConfig.n_layer = out["config"]["n_layer"]
# gpt.GPTConfig.block_size = out["config"]["block_size"]
# gpt.GPTConfig.dropout = out["config"]["dropout"]
# gpt.GPTConfig.vocab_size = out["config"]["vocab_size"]

# t = train.train(32)
# # t.from_pretrained(out["state_dict"])
# t.prepare(data, 1)
# out = t.train(4e-3, 1000, checkpoints = {"path": "bin\\models\\checkpoints", "name": "claw", "interval": 2000})
# t.save(out, "bin\\models\\claw")

# i = gpt.inference(out, True)
# dprint(tokenizer.decode(i.generate(tokenizer.encode("Wake up daddy's home\n<|startoftext|>", allowed_special="all"))))
