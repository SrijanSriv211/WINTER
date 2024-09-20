from src.models.gpt import train, sample
from src.tokenizers.bpe.regex import RegexTokenizer
from src.shared.utils import dprint
import torch

tokenizer = RegexTokenizer()
tokenizer.load("bin\\models\\cl2k.model")
# tokenizer.train(raw_data, 2279, verbose=True)
tokenizer.register_special_tokens({"<|clis|>": 2279, "<|startoftext|>": 2280, "<|endoftext|>": 2281})
# tokenizer.save("bin\\models\\cl2k")

CONFIG = dict(
    # checkpoints
    checkpoints = None,

    # data
    gradient_accumulation_steps = 1, # used to simulate larger batch sizes
    batch_size = 32, # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = 50,

    # model
    vocab_size = 2482,
    n_layer = 4,
    n_head = 4,
    n_embd = 32,
    dropout = 0.0, # for pretraining 0 is good, for finetuning try 0.1+
    bias = False, # do we use bias inside LayerNorm and Linear layers?

    # adamw optimizer
    learning_rate = 4e-3, # max learning rate
    weight_decay = 1e-1,
    beta1 = 0.9,
    beta2 = 0.95,
    grad_clip = 1, # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr = True, # whether to decay the learning rate
    warmup_iters = 1000, # how many steps to warm up for
    lr_decay_iters = 10000, # should be ~= max_iters per Chinchilla
    min_lr = 1e-4, # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # system
    device = "cpu", # examples: "cpu", "cuda", "cuda:0", "cuda:1" etc., or try "mps" on macbooks
    seed = "auto", # examples: "auto", 1337 or any other number
    compile = False # use PyTorch 2.0 to compile the model to be faster
)

with open("data\\claw.txt", "r", encoding="utf-8") as f:
    data = tokenizer.encode(f.read(), allowed_special="all")

out = torch.load("bin\\models\\claw.pth", map_location="cpu")
t = train.train(CONFIG)
t.from_scratch()
t.prepare_data(data, 1)
t.plot("bin\\models\\plot\\claw.png")
out = t.train(max_iters=10000, eval_interval=1000, log_interval=500)
torch.save(out, "bin\\models\\claw.pth")

i = sample.sample(out, auto_load=True)
dprint(tokenizer.decode(i.generate(tokenizer.encode("Wake up WINTER daddy's home\n<|startoftext|>"))))
