from ..shared.utils import calc_total_time
from colorama import Style, Fore, init
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.nn as nn, torch
import time, os

init(autoreset=True)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#                       __  __  ____  _____  ______ _      
#                      |  \/  |/ __ \|  __ \|  ____| |     
#                      | \  / | |  | | |  | | |__  | |     
#                      | |\/| | |  | | |  | |  __| | |     
#                      | |  | | |__| | |__| | |____| |____ 
#                      |_|  |_|\____/|_____/|______|______|
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class RNNConfig:
    input_size: int = None
    output_size: int = None
    n_hidden: int = 4
    n_embd: int = 8
    n_layer: int = 4
    dropout: int = 0
    bias: bool = True

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        assert config.input_size is not None
        assert config.output_size is not None
        self.config = config

        # Defining the layers
        self.embedding = nn.Embedding(config.input_size, config.n_embd) if config.n_embd != None else None # Embedding Layer
        self.rnn = nn.RNN(config.n_embd, config.n_hidden, config.n_layer, bias=config.bias, batch_first=True, dropout=config.dropout) # RNN Layer
        self.fc = nn.Linear(config.n_hidden, config.output_size) # Fully connected layer

    def forward(self, x, targets=None):
        # Pass input through the embedding layer
        if (self.embedding != None):
            x = self.embedding(x)

        # Apply RNN layer
        out, _ = self.rnn(x)

        # Squeeze the dimensions to remove the batch_first dimension
        out = self.fc(out)

        # Calculate loss
        loss = None if targets is None else F.cross_entropy(out, targets)
        return out, loss

    def predict(self, x):
        out, _ = self(x)
        _, predicted = torch.max(out, dim=1)

        probs = torch.softmax(out, dim=1)
        prob = probs[0][predicted.item()]
        confidence = prob.item()

        return predicted.item(), confidence

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#                    _____         __  __ _____  _      ______ 
#                   / ____|  /\   |  \/  |  __ \| |    |  ____|
#                  | (___   /  \  | \  / | |__) | |    | |__   
#                   \___ \ / /\ \ | |\/| |  ___/| |    |  __|  
#                   ____) / ____ \| |  | | |    | |____| |____ 
#                  |_____/_/    \_\_|  |_|_|    |______|______|
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class sample:
    def __init__(self, checkpoint, device="auto", auto_load=True):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

        self.state_dict = checkpoint["model"]
        self.gptconf = RNNConfig(**checkpoint["hyperparams"])

        # automatically load the model
        if auto_load: self.load()

    def load(self):
        # create an instance of RNN
        self.model = RNN(self.gptconf)

        # load the saved model state_dict
        self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()  # set the model to evaluation mode

    # use the model for classification or other tasks
    def predict(self, X, classes):
        i, confidence = self.model.predict(self.prepare_context(X))
        tag = classes[i]

        return tag, confidence

    def prepare_context(self, X):
        X = X.reshape(1, X.shape[0])
        return torch.tensor(X).to(RNNConfig.device)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#                   _______ _____            _____ _   _ 
#                  |__   __|  __ \     /\   |_   _| \ | |
#                     | |  | |__) |   /  \    | | |  \| |
#                     | |  |  _  /   / /\ \   | | | . ` |
#                     | |  | | \ \  / ____ \ _| |_| |\  |
#                     |_|  |_|  \_\/_/    \_\_____|_| \_|
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class train:
    def __init__(self, config):
        """
        `config`: configuration for how to setup the trainer
        ```python
        dict(
            # checkpoints
            checkpoints = {
                "path": "directory-to-save-in",
                "name": "model-name",
                "interval": after-how-many-iters-to-save-checkpoint(int)
            },

            # data
            batch_size = 32,

            # model
            input_size = 2482,
            output_size = 2482,
            n_layer = 4,
            n_hidden = 4,
            n_embd = 32,
            dropout = 0.0, # for pretraining 0 is good, for finetuning try 0.1+
            bias = False, # do we use bias inside LayerNorm and Linear layers?

            # adamw optimizer
            learning_rate = 4e-2, # max learning rate

            # system
            device = "cpu", # examples: "cpu", "cuda", "cuda:0", "cuda:1" etc., or try "mps" on macbooks
            seed = "auto", # examples: "auto", 1337 or any other number
        )
        ```
        """

        self.config = config
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if self.config["device"] == "auto" else self.config["device"]

        torch.manual_seed(self.config["seed"]) if self.config["seed"] != "auto" else None

        # a dict for keep track of all the losses to be plotted.
        self.losses = {
            "train": [],
            "val": []
        }

        # print the device
        print("Training on", f"{Fore.YELLOW}{Style.BRIGHT}{self.device}", f"{Fore.WHITE}{Style.BRIGHT}({torch.initial_seed()})")

    def from_scratch(self):
        self.iter_num = 1

        self.hyperparams = dict(dropout=self.config["dropout"])
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ["n_layer", "n_head", "n_embd", "output_size", "bias", "input_size"]:
            self.hyperparams[k] = self.config[k]

        rnnconf = RNNConfig(**self.hyperparams)
        # create an instance of RNN
        self.model = RNN(rnnconf)
        self.model.to(self.device)

    def prepare(self, encoded_data, data_division=1):
        """
        `data`: The encoded training text data.

        For eg,
        ```python
        encode(text, stoi=self.stoi)
        ```
        """

        data = torch.tensor(encoded_data, dtype=torch.long)

        # train and test splits
        n = int(data_division * len(data)) # the first (data_division * 100)% will be train, rest val
        self.train_data = data[:n]
        self.val_data = data[n:] if 0 < data_division < 1 else data[:n]

        # print the number of tokens
        print(f"{Fore.WHITE}{Style.BRIGHT}{(len(data)/1e6)}M", "total tokens")
        print(
            f"{Fore.WHITE}{Style.BRIGHT}{(len(self.train_data)/1e6)}M", "train tokens,",
            f"{Fore.WHITE}{Style.BRIGHT}{(len(self.val_data)/1e6)}M", "test tokens",
            f"   {Fore.WHITE}{Style.DIM}(Using train tokens as test tokens)" if not (0 < data_division < 1) else ""
        )

    # data loading
    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - 1, (self.config["batch_size"],))
        x = torch.stack([torch.tensor(data[i][0]) for i in ix])
        y = torch.stack([torch.tensor(data[i][1]) for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self, eval_iters):
        out = []
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(split)
                _, loss = self.model(X, Y)

                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self, max_iters=1000, eval_interval=100, eval_iters=100):
        # report number of parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"{Fore.WHITE}{Style.BRIGHT}{n_params/1e6}M", "parameters")

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])

        # training loop
        start_time = time.time()
        eval_time = time.time()
        while True:
            try:
                # evaluate the loss on train/val sets and write checkpoints
                if self.iter_num % eval_interval == 0:
                    losses = self.estimate_loss(eval_iters)

                    print(
                        f"{Fore.WHITE}{Style.BRIGHT}step",
                        f"{Fore.BLACK}{Style.BRIGHT}[{self.iter_num}/{max_iters}]"
                        f"{Fore.RESET}{Style.RESET_ALL}:",
                        f"train loss {Fore.WHITE}{Style.BRIGHT}{losses["train"]:.4f}"
                        f"{Fore.RESET}{Style.RESET_ALL},",
                        f"val loss {Fore.WHITE}{Style.BRIGHT}{losses["val"]:.4f}"
                        f"{Fore.RESET}{Style.RESET_ALL},"
                        f"time took {Fore.BLACK}{Style.BRIGHT}{calc_total_time(time.time() - eval_time)}"
                    )

                    self.losses["train"].append(losses["train"])
                    self.losses["val"].append(losses["val"])
                    eval_time = time.time()

                if self.config["checkpoints"] and self.iter_num % self.config["checkpoints"]["interval"] == 0:
                    if not os.path.isdir(self.config["checkpoints"]["path"]):
                        os.mkdir(self.config["checkpoints"]["path"])

                    if self.iter_num > 0:
                        torch.save(self.get_trained_model(), f"{self.config["checkpoints"]["path"]}\\{self.config["checkpoints"]["name"]}_step{self.iter_num}.pth")

                # sample a batch of data
                X, Y = self.get_batch("train")

                # evaluate the loss
                _, loss = self.model(X, Y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                self.iter_num += 1

                # termination conditions
                if self.iter_num > max_iters:
                    break

            except KeyboardInterrupt:
                print(f"{Fore.RED}{Style.BRIGHT}Early stopping.")
                break

        print("Total time:", calc_total_time(time.time() - start_time))
        return self.get_trained_model()

    def get_trained_model(self):
        return {
            "model": self.model.state_dict(),
            "hyperparams": self.hyperparams,
            "iter_num": self.iter_num,
            "device": self.device
        }

    def plot(self, path):
        plt.style.use("seaborn-v0_8-dark")

        for param in ["figure.facecolor", "axes.facecolor", "savefig.facecolor"]:
            plt.rcParams[param] = "#212946"  # bluish dark grey

        for param in ["text.color", "axes.labelcolor", "xtick.color", "ytick.color"]:
            plt.rcParams[param] = "0.9"  # very light grey

        plt.figure(figsize=(18, 8))
        plt.plot(self.losses["train"], label="train loss")
        plt.plot(self.losses["val"], label="val loss")

        plt.xlabel("iteration", fontsize=12)
        plt.ylabel("value", fontsize=12)
        plt.legend(fontsize=12)
        plt.title("train-val loss", fontsize=14)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
