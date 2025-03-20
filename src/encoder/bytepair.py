from ..shared.utils import calc_total_time
from colorama import init, Fore, Style
from collections import Counter
import pickle, regex, time

init(autoreset=True)

# the main GPT text split patterns, see
# https://stackoverflow.com/a/63871635/18121288
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# a few helper functions useful for both Encoder
def get_stats(ids, most_common=-1):
	stats = Counter([
		pair
		for chunk_ids in ids
		for pair in zip(chunk_ids, chunk_ids[1:])
	])

	return dict(stats.most_common(most_common) if most_common > 0 else stats)

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
		self.vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes

	def train(self, filename, vocab_size=256, batch_size=10, do_merge=True):
		"""
		- vocab_size: max number of merges to be made - 256 bytes
		- batch_size: how many merges to be made before replacing all the made merges in encoded sequence
		"""
		self.do_merge = do_merge
		assert vocab_size >= 256
		assert 1 <= batch_size <= vocab_size
		start_time = time.time()

		with open(filename, "r", encoding="utf-8") as f:
			text = f.read()

		print(
			"encoding text with", f"{Fore.WHITE}{Style.BRIGHT}{len(text)/1e6}M", "characters and",
			f"{Fore.WHITE}{Style.BRIGHT}{len(set(text))}", "unique characters"
		)

		# here are all the unique word that occur in this text
		f256_chr = [chr(idx) for idx in range(256)] # first 256 chars
		t_chr = sorted(list(set(list(text) + f256_chr))) # all text chars
		lt_chr = len(t_chr) # len of all text chars
		text_chunks = regex.findall(self.compiled_pattern, text) # text tokenized by regex
		del f256_chr, text

		print(f"encoding text chunks... {Fore.BLACK}{Style.BRIGHT}(takes a ~minute)")

		if self.do_merge:
			self.vocab = {i: ch.encode("utf-8") for i, ch in enumerate(t_chr)} # word -> idx
			inverse_vocab = {v: k for k, v in self.vocab.items()}
			ids = [[inverse_vocab[i.encode("utf-8")] for i in ch] for ch in text_chunks]

			del inverse_vocab, text_chunks, t_chr
			self._merge_bytepairs(ids, vocab_size, batch_size, lt_chr)

		else:
			text_chunks = list(set(text_chunks + t_chr))
			self.vocab = {i: ch.encode("utf-8") for i, ch in enumerate(text_chunks)} # word -> idx

		# print the total time taken to do all the merges
		print("vocab size:", f"{Fore.WHITE}{Style.BRIGHT}{len(self.vocab)}")
		print("time taken:", f"{Fore.WHITE}{Style.BRIGHT}{calc_total_time(time.time()-start_time)}")

	def _merge_bytepairs(self, ids, vocab_size, batch_size, lt_chr):
		print("training on vocab size", f"{Fore.WHITE}{Style.BRIGHT}{vocab_size}")
		last_print_time = time.time()

		# count the number of times every consecutive pair appears
		i = 0
		merges = {}
		n_merges = vocab_size - lt_chr
		while i < n_merges:
			# passing in stats will update it in place, adding up counts
			# get the pairs with the highest counts
			for pair, occurrences in get_stats(ids, most_common=batch_size).items():
				# mint a new token: assign it the next available id
				idx = lt_chr + i

				# save the merge
				merges[pair] = idx
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
			ids = [merge(chunk_ids, merges) for chunk_ids in ids]

	# special_tokens is a dictionary of str -> int
	# example: {"<|endoftext|>": 100257}
	def register_special_tokens(self, *special_tokens):
		self.special_tokens = dict([(x, i + len(self.vocab)) for i, x in enumerate(special_tokens)])
		self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

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
		return text_bytes.decode("utf-8", errors="replace")

	# return the token ids
	# let's begin. first, convert all bytes to integers in range 0..255
	def _encode_chunk(self, text_bytes, inverse_vocab):
		ids = list(text_bytes)
		if len(ids) < 2:
			return ids

		first, last = 0, 2

		while first <= len(ids):
			if len(ids[first:last]) < 2:
				break

			i0 = ids[first:last][0]
			i1 = ids[first:last][1]

			if self.vocab[i0] + self.vocab[i1] in inverse_vocab.keys():
				ids[first:last] = [inverse_vocab[self.vocab[i0] + self.vocab[i1]]]
				first, last = 0, 2

			else:
				first += 1
				last += 1

		return ids

	def encode_ordinary(self, text):
		"""Encoding that ignores any special tokens."""
		# split text into chunks of text by categories defined in regex pattern
		text_chunks = regex.findall(self.compiled_pattern, text)

		# all chunks of text are encoded separately, then results are joined
		ids = []

		inverse_vocab = {v: k for k, v in self.vocab.items()}
		for chunk in text_chunks:
			chunk_bytes = chunk.encode("utf-8") # raw bytes

			if self.do_merge:
				chunk_ids = self._encode_chunk(chunk_bytes, inverse_vocab)
				ids.extend(chunk_ids)
				continue

			# if self.do_merge == False
			if chunk_bytes in inverse_vocab:
				ids.append(inverse_vocab[chunk_bytes])

			else:
				for ch in list(chunk_bytes.decode()):
					ids.append(inverse_vocab[ch.encode("utf-8")] if ch.encode("utf-8") in inverse_vocab else len(inverse_vocab) + len(self.special_tokens) + 10)

		return ids

	def encode(self, text, allowed_special="none_raise", do_merge=True):
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
				"pattern": self.pattern,
				"special": self.special_tokens,
				"vocab": self.vocab,
				"do_merge": self.do_merge
			}, f)

	def load(self, checkpoint: str):
		# read the model file
		with open(checkpoint, "rb") as f:
			model = pickle.load(f)

		self.pattern = model["pattern"]
		self.special_tokens = model["special"]
		self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
		self.vocab = model["vocab"]
		self.do_merge = model["do_merge"]
