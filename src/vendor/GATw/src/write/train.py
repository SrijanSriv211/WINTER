from ..models.GPT import GPTConfig, GPT
from ..utils import encode
import matplotlib.pyplot as plt
import torch, time, os

class Train:
    def __init__(self, n_layer, n_embd, n_head, lr, dropout, block_size, batch_size, device="auto"):
        # hyperparameters
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.learning_rate = lr
        self.dropout = dropout
        self.block_size = block_size # what is the maximum context length for predictions?
        self.batch_size = batch_size # how many independent sequences will we process in parallel?
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

        # a dict for keep track of all the losses to be plotted.
        self.losses = {
            "train": [],
            "val": []
        }

        # print the device
        print("Training on", self.device)

    def preprocess(self, filepath, data_division=0.8):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        # Train and test splits
        data = torch.tensor(encode(text, stoi=self.stoi), dtype=torch.long)
        n = int(data_division * len(data)) # the first (data_division * 100)% will be train, rest val
        self.train_data = data[:n]
        self.val_data = data[n:]

        # print the number of tokens
        print(len(data)/1e6, "M total tokens")
        print(len(self.train_data)/1e6, "M train tokens,", len(self.val_data)/1e6, "M test tokens")

    # data loading
    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
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

    def train(self, n_steps, eval_interval, eval_iters, checkpoint_interval=0, checkpoint_path="", n_loss_digits=4):
        """
        @param n_steps: number of Epochs to train the model for
        @param eval_interval: the interval between each loss evaluation
        @param eval_iters: the iterations for each loss evaluation
        @param checkpoint_interval: the interval between each checkpoint save
        @param checkpoint_path: the save path for the checkpoint
        @param n_loss_digits: Number of digits of train and val loss printed (default: 4)
        """

        # Set hyperparameters
        GPTConfig.n_embd = self.n_embd
        GPTConfig.n_head = self.n_head
        GPTConfig.n_layer = self.n_layer
        GPTConfig.block_size = self.block_size
        GPTConfig.dropout = self.dropout
        GPTConfig.vocab_size = self.vocab_size
        GPTConfig.device = self.device

        # Create an instance of GPT
        self.model = GPT()
        m = self.model.to(self.device)
        # print the number of parameters in the model
        print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # start timer
        start_time = time.perf_counter()

        # train the model for n_steps
        for iter in range(n_steps):
            try:
                if (iter + 1) % eval_interval == 0 or iter == n_steps - 1:
                    losses = self.estimate_loss(eval_iters)
                    print(f"step [{iter + 1}/{n_steps}]: train loss {losses['train']:.{n_loss_digits}f}, val loss {losses['val']:.{n_loss_digits}f}")
                    self.losses["train"].append(losses['train'])
                    self.losses["val"].append(losses['val'])

                # sample a batch of data
                xb, yb = self.get_batch('train')

                # evaluate the loss
                logits, loss = self.model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if checkpoint_interval != 0 and checkpoint_path != "" and (iter + 1) % checkpoint_interval == 0:
                    # split the filepath into path, filename, and extension
                    path, filename_with_extension = os.path.split(checkpoint_path)
                    filename, extension = os.path.splitext(filename_with_extension)

                    # save the model checkpoint
                    self.save(os.path.join(path, f"{filename}_{(iter + 1)}{extension}"))

            except KeyboardInterrupt:
                break

        print(f"Time taken: {(time.perf_counter() - start_time):.0f} sec")

    def save(self, savepath):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "stoi": self.stoi,
                "itos": self.itos,
                "device": self.device,
                "config": {
                    "n_embd": self.n_embd,
                    "n_head": self.n_head,
                    "n_layer": self.n_layer,
                    "block_size": self.block_size,
                    "dropout": self.dropout,
                    "vocab_size": self.vocab_size
                }
            },
            savepath
        )

    def plot(self, savepath):
        plt.style.use("seaborn-v0_8-dark")

        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = '#212946'  # bluish dark grey

        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0.9'  # very light grey

        plt.figure(figsize=(18, 8))
        plt.plot(self.losses["train"], label="train loss")
        plt.plot(self.losses["val"], label="val loss")

        plt.xlabel("iteration", fontsize=12)
        plt.ylabel("value", fontsize=12)
        plt.legend(fontsize=12)
        plt.title("train-val loss", fontsize=14)
        plt.savefig(savepath, bbox_inches="tight")
        plt.close()
