from colorama import init, Fore, Style
from ..shared.utils import calc_total_time
from collections import Counter
import unicodedata, time
import regex as re

init(autoreset=True)

class RegexTokenizer:
	def __init__(self, pattern=None):
		"""
		- pattern: optional string to override the default (GPT-4 split pattern)
		- special_tokens: str -> int dictionary of special tokens
		  example: {'<|endoftext|>': 100257}
		"""

		self.pattern = pattern
		self.special_tokens = {}
		self.inverse_special_tokens = {}

	def train(self, text, vocab_size):
		"""
		- text: dataset to train the tokenizer on
		- vocab size: size of the vocabulary dictionary of tokens
		- verbose: log output or not
		"""

		assert vocab_size >= 256

		print("encoding text with", f"{Fore.WHITE}{Style.BRIGHT}{len(text)/1e6}M", "characters and", f"{Fore.WHITE}{Style.BRIGHT}{len(set(text))}", "unique characters")
		text_chunks: list[str] = self.__gen_text_chunks__(text)
		ids = [list(ch.encode("utf-8")) for ch in text_chunks]
		del text_chunks

		# iteratively merge the most common pairs to create new tokens
		merges = {} # (int, int) -> int
		vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes

		print("training on vocab size", f"{Fore.WHITE}{Style.BRIGHT}{vocab_size}")
		start_time = time.time()
		last_print_time = time.time()

		# count the number of times every consecutive pair appears
		i = 0
		n_merges = vocab_size - 256
		while i < n_merges:
			# passing in stats will update it in place, adding up counts
			stats = Counter([pair for chunk_ids in ids for pair in zip(chunk_ids, chunk_ids[1:])]).most_common(10)
			# get the pairs with the highest counts
			for pair, occurences in stats:
				# mint a new token: assign it the next available id
				idx = 256 + i

				# save the merge
				merges[pair] = idx
				vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

				# verbose
				print(
					f"{Fore.WHITE}{Style.BRIGHT}merge",
					f"{Fore.BLACK}{Style.BRIGHT}[{i+1}/{n_merges}]"
					":",
					f"{Fore.BLACK}{Style.BRIGHT}{pair} -> {idx}",
					f"{Fore.WHITE}{Style.DIM}({vocab[idx]})",
					f"had {Fore.WHITE}{Style.BRIGHT}{occurences}{Style.RESET_ALL} occurrences"
					f"{Style.RESET_ALL},",
					f"{Fore.BLACK}{Style.BRIGHT}time taken: {calc_total_time(time.time()-last_print_time)}"
				)
				last_print_time = time.time()
				i += 1

			# replace all occurrences of pair in ids with idx
			ids = [self.__merge__(chunk_ids, merges) for chunk_ids in ids]

		print("time taken: ", f"{Fore.WHITE}{Style.BRIGHT}{calc_total_time(time.time()-start_time)}")

		# save class variables
		self.merges = merges # used in encode()
		self.vocab = vocab   # used in decode()

	# split the text up into text chunks using regex pattern
	def __gen_text_chunks__(self, text):
		# the main GPT text split patterns, see
		# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
		GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
		pattern = GPT4_SPLIT_PATTERN if self.pattern is None else self.pattern
		compiled_pattern = re.compile(pattern)

		return re.findall(compiled_pattern, text)

	def __merge__(self, ids, merges: dict):
		"""
		In the list of integers (ids), replace all consecutive occurrences of pair with the new integer token idx
		"""

		new_ids = []
		i = 0

		while i < len(ids):
			# if not at the very last position AND the pair matches, replace it
			if i < len(ids) - 1 and merges.get((ids[i], ids[i+1])) != None:
				new_ids.append(merges.get((ids[i], ids[i+1])))
				i += 2

			else:
				new_ids.append(ids[i])
				i += 1

		return new_ids

	# special_tokens is a dictionary of str -> int
	# example: {"<|endoftext|>": 100257}
	def register_special_tokens(self, special_tokens):
		self.special_tokens = special_tokens
		self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

	# given ids (list of integers), return Python string
	def decode(self, ids):
		part_bytes = []
		for idx in ids:
			if idx in self.vocab:
				part_bytes.append(self.vocab[idx])

			elif idx in self.inverse_special_tokens:
				part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))

			else:
				raise ValueError(f"invalid token id: {idx}")

		text_bytes = b"".join(part_bytes)
		text = text_bytes.decode("utf-8", errors="replace")
		return text

	# return the token ids
	# let's begin. first, convert all bytes to integers in range 0..255
	def _encode_chunk(self, text_bytes):
		ids = list(text_bytes)

		# find the pair with the lowest merge index
		while len(ids) >= 2:
			stats = get_stats(ids)
			pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

			# subtle: if there are no more merges available, the key will
			# result in an inf for every single pair, and the min will be
			# just the first pair in the list, arbitrarily
			# we can detect this terminating case by a membership check

			if pair not in self.merges:
				break # nothing else can be merged anymore

			# otherwise let's merge the best pair (lowest merge index)
			idx = self.merges[pair]
			ids = merge(ids, pair, idx)

		return ids

	def encode_ordinary(self, text):
		"""Encoding that ignores any special tokens."""
		# split text into chunks of text by categories defined in regex pattern
		text_chunks = re.findall(self.compiled_pattern, text)

		# all chunks of text are encoded separately, then results are joined
		ids = []

		for chunk in text_chunks:
			chunk_bytes = chunk.encode("utf-8") # raw bytes
			chunk_ids = self._encode_chunk(chunk_bytes)
			ids.extend(chunk_ids)

		return ids

	def encode(self, text, allowed_special="none_raise"):
		"""
		Unlike encode_ordinary, this function handles special tokens.
		allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
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
		# we can use re.split for this. note that surrounding the pattern with ()
		# makes it into a capturing group, so the special tokens will be included
		special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
		special_chunks = re.split(special_pattern, text)

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

	# vocab is simply and deterministically derived from merges
	def _build_vocab(self):
		vocab = {idx: bytes([idx]) for idx in range(256)}

		for (p0, p1), idx in self.merges.items():
			vocab[idx] = vocab[p0] + vocab[p1]

		for special, idx in self.special_tokens.items():
			vocab[idx] = special.encode("utf-8")

		return vocab

	def save(self, file_prefix):
		"""
		Saves two files: file_prefix.vocab and file_prefix.model
		This is inspired (but not equivalent to!) sentencepiece's model saving:
		- model file is the critical one, intended for load()
		- vocab file is just a pretty printed version for human inspection only
		"""
		# write the model: to be used in load() later
		model_file = file_prefix + ".model"
		with open(model_file, 'w') as f:
			# write the version, pattern and merges, that's all that's needed
			f.write(f"{self.pattern}\n")
			# write the special tokens, first the number of them, then each one
			f.write(f"{len(self.special_tokens)}\n")
			for special, idx in self.special_tokens.items():
				f.write(f"{special} {idx}\n")

			# the merges dict
			for idx1, idx2 in self.merges:
				f.write(f"{idx1} {idx2}\n")

		# write the vocab: for the human to look at
		vocab_file = file_prefix + ".vocab"
		inverted_merges = {idx: pair for pair, idx in self.merges.items()}

		with open(vocab_file, "w", encoding="utf-8") as f:
			for idx, token in self.vocab.items():
				# note: many tokens may be partial utf-8 sequences
				# and cannot be decoded into valid strings. Here we're using
				# errors='replace' to replace them with the replacement char �.
				# this also means that we couldn't possibly use .vocab in load()
				# because decoding in this way is a lossy operation!
				s = self.__render_token__(token)
				# find the children of this token, if any
				if idx in inverted_merges:
					# if this token has children, render it nicely as a merge
					idx0, idx1 = inverted_merges[idx]
					s0 = self.__render_token__(self.vocab[idx0])
					s1 = self.__render_token__(self.vocab[idx1])
					f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")

				else:
					# otherwise this is leaf token, just print it
					# (this should just be the first 256 tokens, the bytes)
					f.write(f"[{s}] {idx}\n")

	def load(self, model_file):
		"""Inverse of save() but only for the model file"""
		assert model_file.endswith(".model")

		# read the model file
		merges = {}
		special_tokens = {}
		idx = 256

		with open(model_file, 'r', encoding="utf-8") as f:
			# read the pattern
			self.pattern = f.readline().strip()
			# read the special tokens
			num_special = int(f.readline().strip())
			for _ in range(num_special):
				special, special_idx = f.readline().strip().split()
				special_tokens[special] = int(special_idx)

			# read the merges
			for line in f:
				idx1, idx2 = map(int, line.split())
				merges[(idx1, idx2)] = idx
				idx += 1

		self.merges = merges
		self.special_tokens = special_tokens
		self.vocab = self._build_vocab()

	# first two helper functions...
	# we don't want to print control characters
	# which distort the output (e.g. \n or much worse)
	# https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
	# http://www.unicode.org/reports/tr44/#GC_Values_Table
	def __replace_control_characters__(self, s: str) -> str:
		chars = []

		for ch in s:
			if unicodedata.category(ch)[0] != "C":
				chars.append(ch) # this character is ok

			else:
				chars.append(f"\\u{ord(ch):04x}") # escape

		return "".join(chars)

	# pretty print a token, escaping control characters
	def __render_token__(self, t: bytes) -> str:
		s = t.decode('utf-8', errors='replace')
		s = self.__replace_control_characters__(s)
		return s
