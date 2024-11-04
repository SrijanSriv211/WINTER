from src.tokenizers.bpe import RegexTokenizer
from src.models.gpt import sample
from src.shared.utils import dprint
import torch, json, os

tokenizer = RegexTokenizer()
tokenizer.load("bin\\cl4k.model")
# tokenizer.train("data\\tok_data.txt", 4279, batch_size=100, is_file=True)
# tokenizer.register_special_tokens({"<|pad|>": 4279, "<|sot|>": 4280, "<|eot|>": 4281})
# tokenizer.save("bin\\cl4k")

# CONFIG = dict(
#     # checkpoints
#     checkpoints = {
#         "path": "bin\\ck",
#         "name": "claw",
#         "interval": 200
#     },

#     # data
#     gradient_accumulation_steps = 8, # used to simulate larger batch sizes
#     batch_size = 16, # if gradient_accumulation_steps > 1, this is the micro-batch size
#     block_size = 128,

#     # model
#     vocab_size = 4282,
#     n_layer = 8,
#     n_head = 8,
#     n_embd = 512,
#     dropout = 0, # for pretraining 0 is good, for finetuning try 0.1+
#     bias = False, # do we use bias inside LayerNorm and Linear layers?

#     # adamw optimizer
#     learning_rate = 3e-3, # max learning rate
#     weight_decay = 1e-1,
#     beta1 = 0.9,
#     beta2 = 0.95,
#     grad_clip = 1, # clip gradients at this value, or disable if == 0.0

#     # learning rate decay settings
#     decay_lr = True, # whether to decay the learning rate
#     warmup_iters = 1000, # how many steps to warm up for
#     lr_decay_iters = 5000, # should be ~= max_iters per Chinchilla
#     min_lr = 3e-5, # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

#     # system
#     device = "cpu",
#     seed = "auto", # examples: "auto", 1337 or any other number
#     compile = True # use PyTorch 2.0 to compile the model to be faster
# )

# t = train(CONFIG)
# t.from_scratch()
# # t.from_pretrained(torch.load("bin\\ck\\claw.pth"))
# t.get_data("data\\train", "data\\val", is_file=False)
# out = t.train(max_iters=5000, eval_interval=200, log_interval=10)
# torch.save(out, "bin\\claw.pth")

out = torch.load("bin\\claw.bin")
i = sample()
i.load(out)

# test = [
#     "Wake up WINTER daddy's home\n<|sot|>",
#     "Hi buddy!\n<|sot|>",
#     "<|sot|>"
# ]

# for text in test:
#     dprint(tokenizer.decode(i.generate(tokenizer.encode(text, allowed_special="all"))))
dprint(tokenizer.decode(i.generate()))

"""
from src.shared.utils import dprint
from src.core.llm import GROQ

llm = GROQ(
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
