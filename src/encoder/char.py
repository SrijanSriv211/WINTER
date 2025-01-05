from ..shared.utils import calc_total_time
from colorama import init, Fore, Style
import pickle, regex, time

init(autoreset=True)

class Encoder:
	def __init__(self):
		self.special_tokens = {}
		self.inverse_special_tokens = {}

	def train(self, text, is_file=False):
		start_time = time.time()

		if is_file:
			with open(text, "r", encoding="utf-8") as f:
				text = f.read()

		print(
			"encoding text with", f"{Fore.WHITE}{Style.BRIGHT}{len(text)/1e6}M", "characters and",
			f"{Fore.WHITE}{Style.BRIGHT}{len(set(text))}", "unique characters"
		)

		# here are all the unique characters that occur in this text
		chars = sorted(list(set(text)))

		# create a mapping from characters to integers
		self.vocab = { ch: i for i, ch in enumerate(chars) } # char -> idx

	# special_tokens is a dictionary of str -> int
	# example: {"<|endoftext|>": 100257}
	def register_special_tokens(self, special_tokens: dict):
		self.special_tokens = special_tokens
		self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

	def decode(self, ids):
		part_bytes = []
		inverse_sorted_vocab = {v: k for k, v in self.vocab.items()}

		for idx in ids:
			if idx in inverse_sorted_vocab:
				part_bytes.append(inverse_sorted_vocab[idx])

			elif idx in self.inverse_special_tokens:
				part_bytes.append(self.inverse_special_tokens[idx])

			else:
				raise ValueError(f"invalid token id: {idx}")
			
		return "".join(part_bytes)

	def encode_ordinary(self, text):
		# all chunks of text are encoded separately, then results are joined
		ids = []

		for char in text:
			if char in self.vocab:
				ids.append(self.vocab[char])

			else:
				ids.append(len(self.vocab) + len(self.special_tokens) + 10) # unknown chars that are not present in vocab

		return ids

	def encode(self, text, allowed_special="none_raise"):
		"""
		Unlike encode_ordinary, this function handles special tokens.
		- allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
		  if none_raise, then an error is raised if any special token is encountered in text
		  this is the default tiktoken behavior right now as well
		  any other behavior is either annoying, or a major footgun
		"""

		# decode the user desire w.r.t. handling of special tokens
		special = None

		if allowed_special == "all":
			special = self.special_tokens

		elif allowed_special == "none":
			special = {}

		elif allowed_special == "none_raise":
			special = {}
			assert all(token not in text for token in self.special_tokens)

		elif isinstance(allowed_special, set):
			special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}

		else:
			raise ValueError(f"allowed_special={allowed_special} not understood")
		
		# shortcut: if no special tokens, just use the ordinary encoding
		if not special:
			return self.encode_ordinary(text)

		# otherwise, we have to be careful with potential special tokens in text
		# we handle special tokens by splitting the text
		# based on the occurrence of any exact match with any of the special tokens
		# we can use regex.split for this. note that surrounding the pattern with ()
		# makes it into a capturing group, so the special tokens will be included
		special_pattern = "(" + "|".join(regex.escape(k) for k in special) + ")"
		special_chunks = regex.split(special_pattern, text)

		# now all the special characters are separated from the rest of the text
		# all chunks of text are encoded separately, then results are joined
		ids = []
		for part in special_chunks:
			# this is a special token, encode it separately as a special case
			if part in special:
				ids.append(special[part])

			# this is an ordinary sequence, encode it normally
			else:
				ids.extend(self.encode_ordinary(part))

		return ids

	def save(self, checkpoint):
		"""
		Saves two files: checkpoint.bin
		- model file is the critical one, intended for load()
		"""
		# write the model: to be used in load() later
		with open(checkpoint, "wb") as f:
			pickle.dump({
				"special": self.special_tokens,
				"vocab": self.vocab
			}, f)

	def load(self, checkpoint: str):
		# read the model file
		with open(checkpoint, "rb") as f:
			model = pickle.load(f)

		self.special_tokens = model["special"]
		self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
		self.vocab = model["vocab"]
