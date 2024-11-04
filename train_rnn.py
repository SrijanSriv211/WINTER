from src.shared.utils import calc_total_time
from src.models.rnn import RNNConfig, RNN
from colorama import Style, Fore, init
import torch._inductor.config as config
import pickle, random, torch, json, time, os

init(autoreset=True)

with open("res\\config.json", "r", encoding="utf-8") as f:
	CONFIG = json.load(f)["RNN"]

device = ("cuda" if torch.cuda.is_available() else "cpu") if CONFIG["device"] == "auto" else CONFIG["device"]

# init seed
torch.manual_seed(CONFIG["seed"]) if CONFIG["seed"] != "auto" else None
random.seed(CONFIG["seed"]) if CONFIG["seed"] != "auto" else None

# print the device
print("Training on", f"{Fore.YELLOW}{Style.BRIGHT}{device}", f"{Fore.WHITE}{Style.BRIGHT}({torch.initial_seed()})")

hyperparams = dict(dropout=CONFIG["dropout"])
# read off the created config params, so we can store them into checkpoint correctly
for k in ["input_size", "output_size", "n_layer", "n_hidden", "dropout"]:
	hyperparams[k] = CONFIG[k]

rnnconf = RNNConfig(**hyperparams)
# create an instance of RNN
model = RNN(rnnconf)
model.to(device)

# optimizer
optimizer = model.configure_optimizers(CONFIG["learning_rate"], CONFIG["device"])

# a dict for keep track of all the losses to be plotted.
metrics = {
	"train": [],
	"eval": [],
	"val": []
}
iter_num = 0
best_loss = 0

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

	ix = torch.randint(len(data) - 1, (CONFIG["batch_size"],))
	x = torch.stack([data[i][0] for i in ix]).float()
	y = torch.stack([data[i][1] for i in ix])
	x, y = x.to(device), y.to(device)
	return x, y

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(eval_iters):
	out = []
	model.eval()
	for split in ["train", "val"]:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			X, Y = get_batch(split)
			logits, loss = model(X, Y)

			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

# report number of parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"{Fore.WHITE}{Style.BRIGHT}{n_params/1e6}M", "parameters")

# compile the model
if CONFIG["compile"]:
	print(f"Compiling the model... {Fore.BLACK}{Style.BRIGHT}(takes a ~minute)")
	#NOTE: backend="inductor" is giving some errors so switched to aot_eager.
	model = torch.compile(model, backend="aot_eager") # requires PyTorch 2.0

# training loop
start_time = time.time()
eval_t0 = time.time()
t0 = time.time()

while True:
	try:
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
				f"time took {Fore.BLACK}{Style.BRIGHT}{calc_total_time(eval_dt)}"
			)

			metrics["train"].append(losses["train"])
			metrics["val"].append(losses["val"])

		if config["checkpoints"] and iter_num % config["checkpoints"]["interval"] == 0:
			if not os.path.isdir(config["checkpoints"]["path"]):
				os.mkdir(config["checkpoints"]["path"])

			if iter_num > 0:
				torch.save(get_trained_model(), f"{config["checkpoints"]["path"]}\\{config["checkpoints"]["name"]}_step{iter_num}.pth")

		# sample a batch of data
		X, Y = get_batch("train")

		# evaluate the loss
		_, loss = model(X, Y)
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

		# timing and logging
		if iter_num % CONFIG["log_interval"] == 0:
			t1 = time.time()
			dt = t1 - t0
			t0 = t1

			# get loss as float. note: this is a CPU-GPU sync point
			lossf = loss.item()

			print(
				f"{Fore.WHITE}{Style.BRIGHT}iter",
				f"{Fore.BLACK}{Style.BRIGHT}[{iter_num}/{CONFIG["max_iters"]}]"
				f"{Fore.RESET}{Style.RESET_ALL}:",
				f"loss {Fore.WHITE}{Style.BRIGHT}{lossf:.4f}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"time took {Fore.BLACK}{Style.BRIGHT}{calc_total_time(dt)}"
			)
			metrics["eval"].append(lossf)

		iter_num += 1

		# termination conditions
		if iter_num > CONFIG["max_iters"]:
			break

	except KeyboardInterrupt:
		print(f"{Fore.RED}{Style.BRIGHT}Early stopping.")
		break

print("Total time:", calc_total_time(time.time() - start_time))
torch.save(get_trained_model(model, optimizer), CONFIG["save_path"])
