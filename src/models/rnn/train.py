from colorama import init, Fore, Style
from .model import RNNConfig, RNN
import torch, time

init(autoreset=True)

class train:
    def __init__(self, batch_size):
        RNNConfig.device = ("cuda" if torch.cuda.is_available() else "cpu") if RNNConfig.device == "auto" else RNNConfig.device
        self.batch_size = batch_size # how many independent sequences will we process in parallel?

        print("Training on", f"{Fore.YELLOW}{Style.BRIGHT}{RNNConfig.device}") # print the device

    def prepare(self, data):
        """
        `data`: The encoded training text data.

        For eg,
        ```python
        torch.tensor(encode(text, stoi=self.stoi), dtype=torch.long)
        ```
        """

        self.data = data

        # print the number of tokens
        print(f"{Fore.WHITE}{Style.BRIGHT}{(len(data)/1e6)}", "M total tokens")
        print(RNNConfig.input_size, "vocab size,", RNNConfig.output_size, "output size")

    # data loading
    def get_batch(self):
        # generate a small batch of data of inputs x and targets y
        ix = torch.randint(len(self.data) - 1, (self.batch_size,))
        x = torch.stack([torch.tensor(self.data[i][0]) for i in ix])
        y = torch.stack([torch.tensor(self.data[i][1]) for i in ix])
        x, y = x.to(RNNConfig.device), y.to(RNNConfig.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self, eval_iters):
        out = []
        self.model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = self.get_batch()
            _, loss = self.model(X, Y)
            losses[k] = loss.item()
        out = losses.mean()
        self.model.train()
        return out

    # n_steps: Number of epochs to train the model for
    # eval_interval: The interval between each loss evaluation
    # eval_iters: The iterations for each loss evaluation
    def train(self, lr, n_steps = 1000, eval_interval = 100, eval_iters = 100):
        # Create an instance of RNN
        self.model = RNN()
        m = self.model.to(RNNConfig.device)
        print(f"{Fore.WHITE}{Style.BRIGHT}{(sum(p.numel() for p in m.parameters())/1e6)}", 'M parameters') # print the number of parameters in the model

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # start timer
        start_time = time.perf_counter()

        # train the model for n_steps
        self.losses = []
        for iter in range(n_steps):
            try:
                if (iter + 1) % eval_interval == 0 or iter == n_steps - 1:
                    losses = self.estimate_loss(eval_iters)
                    print(f"step [{iter + 1}/{n_steps}]: train loss {losses:.4f}")
                    self.losses.append(losses)

                # sample a batch of data
                xb, yb = self.get_batch()

                # evaluate the loss
                _, loss = self.model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            except KeyboardInterrupt:
                break

        print(f"Time taken: {Fore.BLUE}{Style.BRIGHT}{(time.perf_counter() - start_time):.0f} sec")

        return {
            "state_dict": self.model.state_dict(),
            "device": RNNConfig.device,
            "config": {
                "n_hidden": RNNConfig.n_hidden,
                "n_layer": RNNConfig.n_layer,
                "input_size": RNNConfig.input_size,
                "output_size": RNNConfig.output_size
            }
        }
