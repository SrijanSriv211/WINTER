import sys
sys.path.insert(0, "D:\\Dev Projects\\WINTER")

from src.models.encoder import Encoder
from src.models.gpt import sample
import warnings, torch

# supress pytorch's future warning:
# You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly.
# It is possible to construct malicious pickle data which will execute arbitrary code during unpickling
# (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details).
# In a future release, the default value for `weights_only` will be flipped to `True`.
# This limits the functions that could be executed during unpickling.
# Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`.
# We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file.
# Please open an issue on GitHub for any issues related to this experimental feature.
warnings.filterwarnings("ignore", category=FutureWarning)

s = sample()
s.load(torch.load("bin\\claw.bin"), True)

enc = Encoder()
enc.load("bin\\cl8k.bin")

test = [
	"Can I say that Calcia is really a branch of math or is it something nonsense",
	"Every year the moon is going",
	"o/ The workings of the Undetailed",
	None
]

for i in test:
	enctxt = enc.encode(i, allowed_special="all") if i != None else i
	print(enc.decode(s.generate(enctxt)))
	print("\n\n\n")
