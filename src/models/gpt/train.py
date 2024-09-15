from colorama import init, Fore, Style
from .model import GPTConfig, GPT
from ...shared.utils import calc_total_time
import torch, time, os

init(autoreset=True)

class train:
    # `model`: if not none then resume training a model otherwise train from scratch
    def __init__(self, batch_size):
        GPTConfig.device = ("cuda" if torch.cuda.is_available() else "cpu") if GPTConfig.device == "auto" else GPTConfig.device
        # how many independent sequences will we process in parallel?
        self.batch_size = batch_size

        # a dict for keep track of all the losses to be plotted.
        self.losses = {
            "train": [],
            "val": []
        }

        # print the device
        print("Training on", f"{Fore.YELLOW}{Style.BRIGHT}{GPTConfig.device}")

        # create an instance of GPT
        self.model = GPT()

    # load the saved model state_dict
    def from_pretrained(self, state_dict):
        self.model.load_state_dict(state_dict)

    def prepare(self, encoded_data, data_division):
        """
        1. `data`: The encoded training text data.

        For eg,
        ```python
        encode(text, stoi=self.stoi)
        ```

        2. `data_division`: The first `(data_division * 100)%` will be train, rest val
        """

        data = torch.tensor(encoded_data, dtype=torch.long)

        # train and test splits
        n = int(data_division * len(data)) # the first (data_division * 100)% will be train, rest val
        self.train_data = data[:n]
        self.val_data = data[n:] if 0 < data_division < 1 else data[:n]

        # print the number of tokens
        print(f"{Fore.WHITE}{Style.BRIGHT}{(len(data)/1e6)}", "M total tokens")
        print(f"{Fore.WHITE}{Style.BRIGHT}{(len(self.train_data)/1e6)}", "M train tokens,", f"{Fore.WHITE}{Style.BRIGHT}{(len(self.val_data)/1e6)}", "M test tokens")
        print(f"{Fore.WHITE}{Style.DIM}Using train tokens as test tokens\n") if not (0 < data_division < 1) else None

    # data loading
    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - GPTConfig.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+GPTConfig.block_size] for i in ix])
        y = torch.stack([data[i+1:i+GPTConfig.block_size+1] for i in ix])
        x, y = x.to(GPTConfig.device), y.to(GPTConfig.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self, eval_iters):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    # `n_steps`: number of epochs to train the model for
    # `eval_interval`: the interval between each loss evaluation
    # `eval_iters`: the iterations for each loss evaluation
    # `checkpoints`: checkpoints = {'path': "directory-to-save-in", "name": "model-name", "interval": after-how-many-iters-to-save-checkpoint(int)}
    def train(self, lr, n_steps = 1000, eval_interval = 100, eval_iters = 100, checkpoints={}):
        # print the number of parameters in the model
        m = self.model.to(GPTConfig.device)
        self.n_parameters = sum(p.numel() for p in m.parameters())/1e6
        print(f"{Fore.WHITE}{Style.BRIGHT}{self.n_parameters}", 'M parameters')

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # start timer
        start_time = time.perf_counter()
        eval_time = time.perf_counter()

        # train the model for n_steps
        for iter in range(n_steps):
            try:
                if (iter + 1) % eval_interval == 0 or iter == n_steps - 1:
                    losses = self.estimate_loss(eval_iters)

                    print(
                        f"{Fore.WHITE}{Style.BRIGHT}step "
                        f"{Fore.BLACK}{Style.BRIGHT}[{iter + 1}/{n_steps}]"
                        f"{Fore.RESET}{Style.RESET_ALL}: "
                        f"train loss {Fore.WHITE}{Style.BRIGHT}{losses['train']:.4f}"
                        f"{Fore.RESET}{Style.RESET_ALL}, "
                        f"val loss {Fore.WHITE}{Style.BRIGHT}{losses['val']:.4f}"
                        f"{Fore.RESET}{Style.RESET_ALL}, "
                        f"time took {Fore.BLACK}{Style.BRIGHT}{calc_total_time(time.perf_counter() - eval_time)}"
                    )

                    self.losses["train"].append(losses['train'])
                    self.losses["val"].append(losses['val'])
                    eval_time = time.perf_counter()

                if checkpoints and (iter + 1) % checkpoints["interval"] == 0:
                    if not os.path.isdir(checkpoints["path"]):
                        os.mkdir(checkpoints["path"])

                    self.save(self.get_trained_model(), f"{checkpoints["path"]}\\step{(iter + 1)}_{checkpoints["name"]}")

                # sample a batch of data
                xb, yb = self.get_batch('train')

                # evaluate the loss
                logits, loss = self.model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            except KeyboardInterrupt:
                print()
                break

        print("Total time:", calc_total_time(time.perf_counter() - start_time))
        return self.get_trained_model()

    def get_trained_model(self):
        return {
            "state_dict": self.model.state_dict(),
            "device": GPTConfig.device,
            "config": {
                "n_embd": GPTConfig.n_embd,
                "n_head": GPTConfig.n_head,
                "n_layer": GPTConfig.n_layer,
                "block_size": GPTConfig.block_size,
                "dropout": GPTConfig.dropout,
                "vocab_size": GPTConfig.vocab_size
            }
        }

    def save(self, model, path):
        torch.save(model, f"{path}.pth")
