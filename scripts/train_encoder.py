import sys
sys.path.insert(0, "D:\\Dev Projects\\WINTER")

from src.encoder.bytepair import Encoder
enc = Encoder()

# enc.train("data\\tok_enc.txt", do_merge=False)
# enc.register_special_tokens("<|pad|>", "<|sot|>", "<|eot|>")
# enc.save("bin\\cl7k.bin")
#* set `vocab_size` in `config.json` 7000

# enc.train("data\\tok_enc.txt", 2497, batch_size=128)
# enc.register_special_tokens("<|pad|>", "<|sot|>", "<|eot|>")
# enc.save("bin\\cl2k.bin")
#* set `vocab_size` in `config.json` 2500

enc.train("data\\tok_enc.txt", 1021, batch_size=4)
enc.register_special_tokens("<|pad|>", "<|sot|>", "<|eot|>")
enc.save("bin\\cl1k.bin")
#* set `vocab_size` in `config.json` 1024
