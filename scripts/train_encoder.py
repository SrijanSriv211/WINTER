import sys
sys.path.insert(0, "D:\\Dev Projects\\WINTER")

# from src.encoder.bytepair import Encoder
# enc = Encoder()

# enc.train("data\\tok_enc.txt", 1293, batch_size=100, drop_bounds_after=0, is_file=True)
# enc.register_special_tokens({"<|pad|>": 1293, "<|sot|>": 1294, "<|eot|>": 1295})
# enc.save("bin\\cl1k.bin")
#* set `vocab_size` in `config.json` 1296

# enc.train("data\\tok_enc.txt", 4279, batch_size=200, drop_bounds_after=0, is_file=True)
# enc.register_special_tokens({"<|pad|>": 4279, "<|sot|>": 4280, "<|eot|>": 4281})
# enc.save("bin\\cl4k.bin")
#* set `vocab_size` in `config.json` 4282

# enc.train("data\\tok_enc.txt", 8279, batch_size=100, drop_bounds_after=7279, is_file=True)
# enc.register_special_tokens({"<|pad|>": 8279, "<|sot|>": 8280, "<|eot|>": 8281})
# enc.save("bin\\cl8k.bin")
#* set `vocab_size` in `config.json` 8282

# from src.encoder.char import Encoder
# enc = Encoder()

# enc.train("data\\tok_enc.txt", is_file=True)
# enc.register_special_tokens({"<|pad|>": 134, "<|sot|>": 135, "<|eot|>": 136})
# enc.save("bin\\cl140.bin")
#* set `vocab_size` in `config.json` 137

from src.encoder.word import Encoder
enc = Encoder()

enc.train("data\\tok_enc.txt", is_file=True)
enc.register_special_tokens({"<|pad|>": 6916, "<|sot|>": 6917, "<|eot|>": 6918})
enc.save("bin\\cl7k.bin")
# * set `vocab_size` in `config.json` 6919
