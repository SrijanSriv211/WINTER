import sys
sys.path.insert(0, "D:\\Dev Projects\\WINTER")

from src.encoder.char import Encoder
enc = Encoder()

enc.train("data\\tok_enc.txt", is_file=True)
enc.register_special_tokens({"<|pad|>": 141, "<|sot|>": 142, "<|eot|>": 143})
enc.save("bin\\cl140.bin")

# from src.encoder.bytepair import Encoder
# enc = Encoder()

# enc.train("data\\tok_enc.txt", 8279, batch_size=100, drop_bounds_after=7279, is_file=True)
# enc.register_special_tokens({"<|pad|>": 8279, "<|sot|>": 8280, "<|eot|>": 8281})
# enc.save("bin\\cl8k.bin")
