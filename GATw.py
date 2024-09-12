from src.models.gpt import train, model as gpt
from src.tokenizers.bpe.regex import RegexTokenizer
from src.shared.utils import dprint
import torch

with open("data\\chat.txt", "r", encoding="utf-8") as f:
    data = f.read()

tokenizer = RegexTokenizer()
tokenizer.train(data, 1024)
tokenizer.save("bin\\models\\cl1k")

gpt.GPTConfig.vocab_size = 1024
gpt.GPTConfig.block_size = 100
gpt.GPTConfig.n_layer = 2
gpt.GPTConfig.n_head = 4
gpt.GPTConfig.n_embd = 8
gpt.GPTConfig.device = "cpu"

t = train.train(128)
t.prepare(torch.tensor(tokenizer.encode(data), dtype=torch.long), 1)
out = t.train(1e-3, 50000, checkpoints = {"path": "bin\\models\\checkpoints", "name": "claw20k", "interval": 10000})
t.save(out, "bin\\models\\claw20k")

i = gpt.inference(out, True)
dprint(tokenizer.decode(i.generate(tokenizer.encode("Human: Hello WINTER!\nWINTER: "), 100)))
