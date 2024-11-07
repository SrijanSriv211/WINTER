import sys
sys.path.insert(0, "D:\\Dev Projects\\WINTER")

from src.shared.utils import calc_total_time
from src.models.gpt import GPTConfig, GPT
from colorama import Style, Fore, init
from contextlib import nullcontext
import warnings, pickle, random, time, math, os
import torch._inductor.config as config
import torch.amp, torch, json

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

init(autoreset=True)

with open("scripts\\config.json", "r", encoding="utf-8") as f:
	CONFIG = json.load(f)["GPT"]

device = ("cuda" if torch.cuda.is_available() else "cpu") if CONFIG["device"] == "auto" else CONFIG["device"]

# init seed
torch.manual_seed(CONFIG["seed"]) if CONFIG["seed"] != "auto" else None
random.seed(CONFIG["seed"]) if CONFIG["seed"] != "auto" else None

# "float32", "bfloat16", or "float16", the latter will auto implement a GradScaler
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=ptdtype)

# print the device
print("Training on", f"{Fore.YELLOW}{Style.BRIGHT}{device}", f"{Fore.WHITE}{Style.BRIGHT}({torch.initial_seed()})")

def from_scratch():
	hyperparams = dict(dropout=CONFIG["dropout"])
	# read off the created CONFIG params, so we can store them into checkpoint correctly
	for k in ["n_layer", "n_head", "n_embd", "block_size", "vocab_size"]:
		hyperparams[k] = CONFIG[k]

	gptconf = GPTConfig(**hyperparams)
	# create an instance of GPT
	model = GPT(gptconf)
	model.to(device)

	# optimizer
	optimizer = model.configure_optimizers(CONFIG["weight_decay"], CONFIG["learning_rate"], CONFIG["device"])

	# a dict for keep track of all the losses to be plotted.
	metrics = {
		"train": [],
		"eval": [],
		"val": [],
		"mfu": [],
		"lr": []
	}
	iter_num = 0
	best_loss = 0

	return model, optimizer, hyperparams, iter_num, best_loss, metrics

def from_pretrained(checkpoint):
	# make loading pretrained models backwards compatible with previously trained models
	metrics = [checkpoint[i] for i in ["metrics", "losses"] if i in checkpoint.keys()]
	metrics = {
		"train": [],
		"eval": [],
		"val": [],
		"mfu": [],
		"lr": []
	} if not metrics else metrics[0]
	for i in ["mfu", "lr"]:
		if i not in metrics.keys():
			metrics[i] = []

	# load the state dict and current iteration number of the model
	state_dict = checkpoint["model"]
	iter_num = checkpoint["iter_num"]
	best_loss = checkpoint["best_loss"] if "best_loss" in checkpoint.keys() else 0

	hyperparams = dict(dropout=CONFIG["dropout"])
	# read off the created config params, so we can store them into checkpoint correctly
	for k in ["n_layer", "n_head", "n_embd", "block_size", "vocab_size"]:
		hyperparams[k] = checkpoint["hyperparams"][k]

	gptconf = GPTConfig(**hyperparams)

	# create an instance of GPT
	model = GPT(gptconf)

	# remove `_orig_mod.` prefix from state_dict (if it's there)
	state_dict = checkpoint["model"]
	unwanted_prefix = '_orig_mod.'

	for k, v in list(state_dict.items()):
		if k.startswith(unwanted_prefix):
			state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

	model.load_state_dict(state_dict)
	model.to(device)

	# optimizer
	optimizer = model.configure_optimizers(CONFIG["weight_decay"], CONFIG["learning_rate"], CONFIG["device"])
	optimizer.load_state_dict(checkpoint["optimizer"])

	# crop down the model block size if desired, using model surgery
	if CONFIG["block_size"] < hyperparams["block_size"]:
		model.crop_block_size(CONFIG["block_size"])
		hyperparams["block_size"] = CONFIG["block_size"] # so that the checkpoint will have the right value

	return model, optimizer, hyperparams, iter_num, best_loss, metrics

# init model and optimizer
if CONFIG["init_from"] == "scratch":
	model, optimizer, hyperparams, iter_num, best_loss, metrics = from_scratch()

elif CONFIG["init_from"].startswith("pretrained,"):
	model, optimizer, hyperparams, iter_num, best_loss, metrics = from_pretrained(torch.load(CONFIG["init_from"][11:]))

# load all the files
train_data, val_data = 0, 0
if CONFIG["load_from_file"]:
	# try to load and check all the data
	with open(CONFIG["train_data"], "rb") as f:
		train_data = len(pickle.load(f))

	with open(CONFIG["val_data"], "rb") as f:
		val_data = len(pickle.load(f))

else:
	for i in os.listdir(CONFIG["train_data"]):
		# try to load and check all the data
		with open(f"{CONFIG["train_data"]}\\{i}", "rb") as f:
			train_data += len(pickle.load(f))

	for i in os.listdir(CONFIG["val_data"]):
		with open(f"{CONFIG["val_data"]}\\{i}", "rb") as f:
			val_data += len(pickle.load(f))

data = train_data + val_data

# print the number of tokens
print(f"{Fore.WHITE}{Style.BRIGHT}{(data/1e6)}M", "total tokens")
print(
	f"{Fore.WHITE}{Style.BRIGHT}{(train_data/1e6)}M", "train tokens,",
	f"{Fore.WHITE}{Style.BRIGHT}{(val_data/1e6)}M", "test tokens",
	f"   {Fore.WHITE}{Style.DIM}(Using train tokens as test tokens)" if not (0 < (train_data/data) < 1) else ""
)
del data, train_data, val_data # these are useless vars, delete them

def get_trained_model(model, optimizer):
	return {
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"hyperparams": hyperparams,
		"device": device,
		"iter_num": iter_num,
		"best_loss": best_loss
	}

def _load_data(path):
	if not CONFIG["load_from_file"]:
		files = os.listdir(path)
		random.shuffle(files)

	with open(f"{path}\\{files[0]}" if not CONFIG["load_from_file"] else path, "rb") as f:
		return pickle.load(f)

# data loading
# generate a small batch of data of inputs x and targets y
def get_batch(split):
	# we reload data every batch to avoid a memory leak
	path = CONFIG["train_data"] if split == "train" else CONFIG["val_data"]
	data = _load_data(path)

	ix = torch.randint(len(data) - CONFIG["block_size"], (CONFIG["batch_size"],))
	x = torch.stack([data[i:i+CONFIG["block_size"]] for i in ix])
	y = torch.stack([data[i+1:i+CONFIG["block_size"]+1] for i in ix])
	x, y = x.to(device), y.to(device)
	return x, y

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(eval_iters):
	out = {}
	model.eval()
	for split in ["train", "val"]:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			X, Y = get_batch(split)
			with ctx:
				logits, loss = model(X, Y)

			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
	# 1) linear warmup for warmup_iters steps
	if it < CONFIG["warmup_iters"]:
		return CONFIG["learning_rate"] * it / CONFIG["warmup_iters"]

	# 2) if it > lr_decay_iters, return min learning rate
	if it > CONFIG["lr_decay_iters"]:
		return CONFIG["min_lr"]

	# 3) in between, use cosine decay down to min learning rate
	decay_ratio = (it - CONFIG["warmup_iters"]) / (CONFIG["lr_decay_iters"] - CONFIG["warmup_iters"])

	assert 0 <= decay_ratio <= 1
	coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
	return CONFIG["min_lr"] + coeff * (CONFIG["learning_rate"] - CONFIG["min_lr"])

# report number of parameters
print(f"{Fore.WHITE}{Style.BRIGHT}{model.get_num_params()/1e6}M", "parameters")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=False)

if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
# compile the model
if CONFIG["compile"]:
	print(f"Compiling the model... {Fore.BLACK}{Style.BRIGHT}(takes a ~minute)")
	#NOTE: backend="inductor" is giving some errors so switched to aot_eager.
	model = torch.compile(model, backend="aot_eager") # requires PyTorch 2.0

# training loop
X, Y = get_batch("train") # fetch the very first batch
start_time = time.time()
eval_t0 = time.time()
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
training_loop = True

while training_loop:
	try:
		# determine and set the learning rate for this iteration
		lr = get_lr(iter_num) if CONFIG["decay_lr"] else CONFIG["learning_rate"]
		for param_group in optimizer.param_groups:
			param_group["lr"] = lr
		metrics["lr"].append(lr)

		# evaluate the loss on train/val sets and write checkpoints
		if iter_num % CONFIG["eval_interval"] == 0:
			losses = estimate_loss(CONFIG["eval_iters"])
			# timing and logging
			eval_t1 = time.time()
			eval_dt = eval_t1 - eval_t0
			eval_t0 = eval_t1

			print(
				f"{Fore.WHITE}{Style.BRIGHT}step",
				f"{Fore.BLACK}{Style.BRIGHT}[{iter_num}/{CONFIG["max_iters"]}]"
				f"{Fore.RESET}{Style.RESET_ALL}:",
				f"train loss {Fore.WHITE}{Style.BRIGHT}{losses["train"]:.4f}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"val loss {Fore.WHITE}{Style.BRIGHT}{losses["val"]:.4f}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"lr {Fore.WHITE}{Style.BRIGHT}{lr:.4f}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"time took {Fore.BLACK}{Style.BRIGHT}{calc_total_time(eval_dt)}"
			)

			metrics["train"].append(losses["train"])
			metrics["val"].append(losses["val"])

		# save checkpoint
		if CONFIG["checkpoints"] != None and iter_num % CONFIG["checkpoints"]["interval"] == 0:
			if not os.path.isdir(CONFIG["checkpoints"]["path"]):
				os.mkdir(CONFIG["checkpoints"]["path"])

			if iter_num > 0:
				print(f"saved checkpoint at step {Fore.WHITE}{Style.BRIGHT}{iter_num}")
				torch.save(get_trained_model(model, optimizer), f"{CONFIG["checkpoints"]["path"]}\\{CONFIG["checkpoints"]["name"]}_step{iter_num}.bin")

		# forward backward update, with optional gradient accumulation to simulate larger batch size
		# and using the GradScaler if data type is float16
		for micro_step in range(CONFIG["gradient_accumulation_steps"]):
			with ctx:
				logits, loss = model(X, Y)
				loss = loss / CONFIG["gradient_accumulation_steps"] # scale the loss to account for gradient accumulation

			# immediately async prefetch next batch while model is doing the forward pass on the GPU
			X, Y = get_batch("train")
			# backward pass, with gradient scaling if training in fp16
			scaler.scale(loss).backward()

		# clip the gradient
		if CONFIG["grad_clip"] != 0.0:
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])

		# step the optimizer and scaler if training in fp16
		scaler.step(optimizer)
		scaler.update()

		# flush the gradients as soon as we can, no need for this memory anymore
		optimizer.zero_grad(set_to_none=True)

		# timing and logging
		if iter_num % CONFIG["log_interval"] == 0:
			t1 = time.time()
			dt = t1 - t0
			t0 = t1

			# get loss as float. note: this is a CPU-GPU sync point
			# scale up to undo the division above, approximating the true total loss (exact would have been a sum)
			lossf = loss.item() * CONFIG["gradient_accumulation_steps"]

			if local_iter_num >= 5: # let the training loop settle a bit
				mfu = model.estimate_mfu(CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"], dt)
				running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu

			print(
				f"{Fore.WHITE}{Style.BRIGHT}iter",
				f"{Fore.BLACK}{Style.BRIGHT}[{iter_num}/{CONFIG["max_iters"]}]"
				f"{Fore.RESET}{Style.RESET_ALL}:",
				f"loss {Fore.WHITE}{Style.BRIGHT}{lossf:.4f}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"mfu {Fore.WHITE}{Style.BRIGHT}{running_mfu*100:.2f}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"time took {Fore.BLACK}{Style.BRIGHT}{calc_total_time(dt)}"
			)
			metrics["mfu"].append(running_mfu)
			metrics["eval"].append(lossf)

		iter_num += 1
		local_iter_num += 1

		# termination conditions
		if iter_num > CONFIG["max_iters"]:
			break

	except KeyboardInterrupt:
		print("Type")
		print(f"{Fore.WHITE}{Style.BRIGHT}1. {Fore.BLACK}{Style.BRIGHT}`y` {Style.RESET_ALL}to stop training.")
		print(f"{Fore.WHITE}{Style.BRIGHT}2. {Fore.BLACK}{Style.BRIGHT}`n` {Style.RESET_ALL}to continue training.")
		print(f"{Fore.WHITE}{Style.BRIGHT}3. {Fore.BLACK}{Style.BRIGHT}`s` {Style.RESET_ALL}to save model.")
		print(f"{Fore.WHITE}{Style.BRIGHT}4. {Fore.BLACK}{Style.BRIGHT}`r` {Style.RESET_ALL}to reload config.json.")

		while True:
			inp = input("> ")

			if inp == "y":
				print(f"{Fore.RED}{Style.BRIGHT}Early stopping.")
				training_loop = False
				break

			elif inp == "n":
				print(f"{Fore.GREEN}{Style.BRIGHT}Continue training.")
				break

			elif inp == "s":
				print(f"{Fore.YELLOW}{Style.BRIGHT}Saving model.")
				print("Total time:", calc_total_time(time.time() - start_time))
				torch.save(get_trained_model(model, optimizer), CONFIG["save_path"])

			elif inp == "r":
				print(f"{Fore.YELLOW}{Style.BRIGHT}config.json{Style.RESET_ALL} reloaded.")
				with open("scripts\\config.json", "r", encoding="utf-8") as f:
					CONFIG = json.load(f)["GPT"]

			else:
				print(f"{Fore.RED}{Style.DIM}Wrong option.")

print("Total time:", calc_total_time(time.time() - start_time))
torch.save(get_trained_model(model, optimizer), CONFIG["save_path"])
