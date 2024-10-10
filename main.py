from src.tokenizers.bpe import RegexTokenizer
from src.models.gpt import train, sample
from src.shared.utils import dprint
import torch

raw_data = ""
with open("data\\claw.txt", "r", encoding="utf-8") as f:
    raw_data = f.read()

with open("data\\jokes.txt", "r", encoding="utf-8") as f:
    raw_data = "\n" + f.read()

with open("data\\facts.txt", "r", encoding="utf-8") as f:
    raw_data += "\n" + f.read()

with open("data\\data.txt", "r", encoding="utf-8") as f:
    raw_data += "\n" + f.read()[:25_000_000]

with open("data\\LLM data.txt", "r", encoding="utf-8") as f:
    raw_data += "\n" + f.read()[:25_000_000]

tokenizer = RegexTokenizer()
tokenizer.from_scratch()
tokenizer.train(raw_data, 4279)
tokenizer.register_special_tokens({"<|pad|>": 4279, "<|sot|>": 4280, "<|eot|>": 4281})
tokenizer.save("bin\\cl2k")

# import time
# from src.shared.utils import calc_total_time
# tokenizer.load("bin\\cl2k.model")
# print(len(raw_data))

# CONFIG = dict(
#     # checkpoints
#     checkpoints = {
#         "path": "bin\\ck",
#         "name": "claw",
#         "interval": 1000
#     },

#     # data
#     gradient_accumulation_steps = 2 * 8, # used to simulate larger batch sizes
#     batch_size = 32, # if gradient_accumulation_steps > 1, this is the micro-batch size
#     block_size = 512,

#     # model
#     vocab_size = 2282,
#     n_layer = 4,
#     n_head = 4,
#     n_embd = 32,
#     dropout = 0.0, # for pretraining 0 is good, for finetuning try 0.1+
#     bias = True, # do we use bias inside LayerNorm and Linear layers?

#     # adamw optimizer
#     learning_rate = 3e-4, # max learning rate
#     weight_decay = 1e-1,
#     beta1 = 0.9,
#     beta2 = 0.95,
#     grad_clip = 1, # clip gradients at this value, or disable if == 0.0

#     # learning rate decay settings
#     decay_lr = True, # whether to decay the learning rate
#     warmup_iters = 1000, # how many steps to warm up for
#     lr_decay_iters = 10000, # should be ~= max_iters per Chinchilla
#     min_lr = 3e-5, # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

#     # system
#     device = "cpu", # examples: "cpu", "cuda", "cuda:0", "cuda:1" etc., or try "mps" on macbooks
#     seed = "auto", # examples: "auto", 1337 or any other number
#     compile = False # use PyTorch 2.0 to compile the model to be faster
# )

# t = train(CONFIG)
# t.from_scratch()
# t.prepare_data(tokenizer.encode(raw_data), 1)
# out = t.train(max_iters=10000, eval_interval=2000, log_interval=500)
# t.plot("res\\claw.png")
# torch.save(out, "bin\\claw.pth")

# # out = torch.load("bin\\claw.pth", map_location="cpu")
# i = sample(out)

# test = [
#     "Wake up WINTER daddy's home\n<|sot|>",
#     "Hi buddy!\n<|sot|>",
#     "<|sot|>"
# ]

# for text in test:
#     dprint(tokenizer.decode(i.generate(tokenizer.encode(text, allowed_special="all"))))

"""
from src.shared.utils import dprint
from src.core.llm import LLM

llm = LLM(
    system = "Your name is WINTER (Witty Intelligence with Natural Emotions and Rationality). "
    "As your name suggests you are kind, helpful, witty, intelligent, emotional, empathetic, rational, clever, charming, funny, innocent, cute and curious. "
    "As innocent, cute, wholesome and curious as Wall-E from Pixar's movie Wall-E. "
    "So your responses must reflect all these traits. "
    "You are currently talking to Srijan Srivastava, it me your creator buddy. "
    "You can call me \"sir\" because I made you to work like Tony Stark's JARVIS. LOL! "
    "Your responses must be simple, short, crips and interesting like JARVIS'. "
    "Don't do any unnecessary talking. Also try to be as human as possible and don't be too generic and AI-like. Be human. "
    "Reply with maximum of 20 words in general.",

    GroqAPI_path = "bin\\cache\\GroqAPI.txt",
    conversation_path = "bin\\cache\\converse.txt")

while True:
    try:
        i = input("> ")

        if i.strip() == "":
            continue

        elif i == "exit":
            llm.save_conversation("bin\\cache\\converse.txt")
            break

        out = llm.generate(i)
        dprint(out)

    except Exception as e:
        llm.save_conversation("bin\\cache\\converse.txt")
        break
"""
