from ..shared.utils import calc_total_time
from colorama import init, Fore, Style
import pickle, regex, time

init(autoreset=True)

# the main GPT text split patterns, see
# https://stackoverflow.com/a/63871635/18121288
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class Encoder:
	def __init__(self, pattern=None):
		"""
		- pattern: optional string to override the default (GPT-4 split pattern)
		- special_tokens: int dictionary of special tokens
		  example: {'<|endoftext|>': 100257}
		"""
		self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
		self.compiled_pattern = regex.compile(self.pattern)

		self.special_tokens = {}
		self.inverse_special_tokens = {}

	def train(self, text, is_file=False):
		start_time = time.time()

		if is_file:
			with open(text, "r", encoding="utf-8") as f:
				text = f.read()

		# here are all the unique word that occur in this text
		text_chunks = regex.findall(self.compiled_pattern, text)
		chunks = sorted(list(set(list(text) + [chr(idx) for idx in range(256)] + text_chunks)))

		print(
			"encoding text with", f"{Fore.WHITE}{Style.BRIGHT}{len(text_chunks)/1e6}M", "chunks and",
			f"{Fore.WHITE}{Style.BRIGHT}{len(chunks)}", "unique chunks"
		)
		del text_chunks, text

		# create a mapping from word to integers
		print(f"encoding chunks... {Fore.BLACK}{Style.BRIGHT}(takes a ~minute)")
		self.vocab = { ch: i for i, ch in enumerate(chunks) } # word -> idx

		# print the total time taken to do all the merges
		print("time taken: ", f"{Fore.WHITE}{Style.BRIGHT}{calc_total_time(time.time()-start_time)}")

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
		"""Encoding that ignores any special tokens."""
		# split text into chunks of text by categories defined in regex pattern
		text_chunks = regex.findall(self.compiled_pattern, text)

		# all chunks of text are encoded separately, then results are joined
		ids = []

		for chunk in text_chunks:
			if chunk in self.vocab:
				ids.append(self.vocab[chunk])
				continue

			for ch in list(chunk):
				if ch in self.vocab:
					ids.append(self.vocab[ch])

				else:
					ids.append(len(self.vocab) + len(self.special_tokens) + 10) # unknown chunks that are not present in vocab

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
				"vocab": self.vocab,
				"pattern": self.pattern
			}, f)

	def load(self, checkpoint: str):
		# read the model file
		with open(checkpoint, "rb") as f:
			model = pickle.load(f)

		self.special_tokens = model["special"]
		self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
		self.vocab = model["vocab"]

		self.pattern = model["pattern"]
		self.compiled_pattern = regex.compile(self.pattern)
