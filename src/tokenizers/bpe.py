from ..shared.utils import calc_total_time
from colorama import init, Fore, Style
from collections import Counter
import unicodedata, time
import regex as re

init(autoreset=True)

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# a few helper functions useful for both RegexTokenizer
def get_stats(ids):
	return dict(Counter([pair for chunk_ids in ids for pair in zip(chunk_ids, chunk_ids[1:])]).most_common(10))

def merge(ids, merges: dict):
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

# first two helper functions...
# we don't want to print control characters
# which distort the output (e.g. \n or much worse)
# https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
# http://www.unicode.org/reports/tr44/#GC_Values_Table
def replace_control_characters(s: str) -> str:
	chars = []

	for ch in s:
		if unicodedata.category(ch)[0] != "C":
			chars.append(ch) # this character is ok

		else:
			chars.append(f"\\u{ord(ch):04x}") # escape

	return "".join(chars)

# pretty print a token, escaping control characters
def render_token(t: bytes) -> str:
	s = t.decode('utf-8', errors='replace')
	s = replace_control_characters(s)
	return s

class RegexTokenizer:
	def __init__(self, pattern=None):
		"""
		- pattern optional string to override the default (GPT-4 split pattern)
		- special_tokens: int dictionary of special tokens
		  example: {'<|endoftext|>': 100257}
		"""

		self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
		self.compiled_pattern = re.compile(self.pattern)
		self.resume_training = False

		self.special_tokens = {}
		self.inverse_special_tokens = {}

		self.merges = {} # (int, int) -> int
		self.merge_offset = 256

	def from_scratch(self):
		self.vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes

	def from_pretrained(self, checkpoint):
		self.load(checkpoint)
		self.special_tokens = {}
		self.merge_offset = len(self.vocab)
		self.resume_training = True

	def train(self, text, vocab_size):
		assert vocab_size >= self.merge_offset

		text_chunks: list[str] = re.findall(self.compiled_pattern, text)
		print(
			"encoding text with", f"{Fore.WHITE}{Style.BRIGHT}{len(text)/1e6}M", "characters and",
			f"{Fore.WHITE}{Style.BRIGHT}{len(set(text))}", "unique characters",
			f"{Fore.WHITE}{Style.BRIGHT}{len(set(text_chunks))}", "unique words"
		)
		ids = [self.encode(ch) if self.resume_training else list(ch.encode("utf-8")) for ch in text_chunks]

		print("training on vocab size", f"{Fore.WHITE}{Style.BRIGHT}{vocab_size}")
		start_time = time.time()
		last_print_time = time.time()

		# count the number of times every consecutive pair appears
		i = 0
		n_merges = vocab_size - self.merge_offset
		while i < n_merges:
			# passing in stats will update it in place, adding up counts
			stats = get_stats(ids)
			# get the pairs with the highest counts
			for pair, occurrences in stats.items():
				# mint a new token: assign it the next available id
				idx = self.merge_offset + i

				# save the merge
				self.merges[pair] = idx
				self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

				# verbose
				print(
					f"{Fore.WHITE}{Style.BRIGHT}merge",
					f"{Fore.BLACK}{Style.BRIGHT}[{i+1}/{n_merges}]"
					":",
					f"{pair} -> {idx}",
					f"{Fore.WHITE}{Style.DIM}({self.vocab[idx]})",
					f"had {Fore.WHITE}{Style.BRIGHT}{occurrences}{Style.RESET_ALL} occurrences"
					f"{Style.RESET_ALL},",
					f"{Fore.BLACK}{Style.BRIGHT}time taken: {calc_total_time(time.time()-last_print_time)}"
				)
				last_print_time = time.time()
				i += 1
				if i >= n_merges:
					break

			# replace all occurrences of pair in ids with idx
			ids = [merge(chunk_ids, self.merges) for chunk_ids in ids]

		# print the total time taken to do all the merges
		print("time taken: ", f"{Fore.WHITE}{Style.BRIGHT}{calc_total_time(time.time()-start_time)}")

	# special_tokens is a dictionary of str -> int
	# example: {"<|endoftext|>": 100257}
	def register_special_tokens(self, special_tokens: dict):
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
			stats = get_stats([ids])
			pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

			# subtle: if there are no more merges available, the key will
			# result in an inf for every single pair, and the min will be
			# just the first pair in the list, arbitrarily
			# we can detect this terminating case by a membership check

			if pair not in self.merges:
				break # nothing else can be merged anymore

			# otherwise let's merge the best pair (lowest merge index)
			ids = merge(ids, self.merges)

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
				s = render_token(token)
				# find the children of this token, if any
				if idx in inverted_merges:
					# if this token has children, render it nicely as a merge
					idx0, idx1 = inverted_merges[idx]
					s0 = render_token(self.vocab[idx0])
					s1 = render_token(self.vocab[idx1])
					f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")

				else:
					# otherwise this is leaf token, just print it
					# (this should just be the first 256 tokens, the bytes)
					f.write(f"[{s}] {idx}\n")

	def load(self, checkpoint: str):
		assert checkpoint.endswith(".model")

		# read the model file
		idx = 256
		self.merges = {}
		self.special_tokens = {}

		with open(checkpoint, "r", encoding="utf-8") as f:
			# read the pattern
			self.pattern = f.readline().strip()

			# read the special tokens
			num_special = int(f.readline().strip())
			for _ in range(num_special):
				special, special_idx = f.readline().strip().split()
				self.special_tokens[special] = int(special_idx)

			# read the merges
			for line in f:
				idx1, idx2 = map(int, line.split())
				self.merges[(idx1, idx2)] = idx
				idx += 1

		# vocab is simply and deterministically derived from merges
		self.vocab = {idx: bytes([idx]) for idx in range(256)}

		for (p0, p1), idx in self.merges.items():
			self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

		for special, idx in self.special_tokens.items():
			self.vocab[idx] = special.encode("utf-8")
