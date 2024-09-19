from colorama import init, Fore, Style
from contextlib import nullcontext
from .model import GPTConfig, GPT
from ...shared.utils import calc_total_time
import torch, time, math, os

init(autoreset=True)

class train:
    def __init__(self, config):
        """
        1. `config`: configuration for how to setup the trainer
        ```python
        dict(
            # checkpoints
            checkpoints = {
                "path": "directory-to-save-in",
                "name": "model-name",
                "interval": after-how-many-iters-to-save-checkpoint(int)
            },

            # data
            gradient_accumulation_steps = 5 * 8, # used to simulate larger batch sizes
            batch_size = 12, # if gradient_accumulation_steps > 1, this is the micro-batch size
            block_size = 1024,

            # model
            vocab_size = 2482,
            n_layer = 12,
            n_head = 12,
            n_embd = 768,
            dropout = 0.0, # for pretraining 0 is good, for finetuning try 0.1+
            bias = False, # do we use bias inside LayerNorm and Linear layers?

            # adamw optimizer
            learning_rate = 6e-4, # max learning rate
            max_iters = 600000, # total number of training iterations
            weight_decay = 1e-1,
            beta1 = 0.9,
            beta2 = 0.95,
            grad_clip = 1.0, # clip gradients at this value, or disable if == 0.0

            # learning rate decay settings
            decay_lr = True, # whether to decay the learning rate
            warmup_iters = 2000, # how many steps to warm up for
            lr_decay_iters = 600000, # should be ~= max_iters per Chinchilla
            min_lr = 6e-5, # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

            # system
            device = "cpu", # examples: "cpu", "cuda", "cuda:0", "cuda:1" etc., or try "mps" on macbooks
            compile = True # use PyTorch 2.0 to compile the model to be faster
        )
        ```

        2. `pretrained`: if not none then resume training a model otherwise train from scratch
        ```python
        train(pretrained="bin\\model\\ckpt.pt")
        ```
        """

        self.config = config
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if self.config.device == "auto" else self.device

        # "float32", "bfloat16", or "float16", the latter will auto implement a GradScaler
        dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        self.ctx = nullcontext() if self.device == "cpu" else torch.amp.autocast(device_type=self.device, dtype=ptdtype)

        # a dict for keep track of all the losses to be plotted.
        self.losses = {
            "train": [],
            "val": []
        }

        # print the device
        print(
            "Training on", f"{Fore.YELLOW}{Style.BRIGHT}{self.device}",
            f"   {Fore.WHITE}{Style.BRIGHT}({torch.seed()})"
        )

    def from_scratch(self):
        self.iter_num = 0

        self.hyperparams = dict(dropout=self.config.dropout)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            self.hyperparams[k] = getattr(self.config, k)

        gptconf = GPTConfig(**self.hyperparams)
        # create an instance of GPT
        self.model = GPT(gptconf)
        self.model.to(self.device)

        # optimizer
        self.optimizer = self.model.configure_optimizers(self.config.weight_decay, self.config.learning_rate, (self.config.beta1, self.config.beta2), self.config.device)

    def from_pretrained(self, pretrained_model):
        """
        Load the pretrained model from path, for eg, `from_pretrained("bin\\model\\ckpt.pt")`
        """

        checkpoint = torch.load(pretrained_model, map_location=self.device)
        state_dict = checkpoint["model"]
        self.iter_num = checkpoint["iter_num"]

        self.hyperparams = dict(dropout=self.config.dropout)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            self.hyperparams[k] = getattr(checkpoint["hyperparams"], k)

        gptconf = GPTConfig(**self.hyperparams)

        # create an instance of GPT
        self.model = GPT(gptconf)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        # optimizer
        self.optimizer = self.model.configure_optimizers(self.config.weight_decay, self.config.learning_rate, (self.config.beta1, self.config.beta2), self.config.device)
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # crop down the model block size if desired, using model surgery
        if self.config.block_size < self.hyperparams["block_size"]:
            self.model.crop_block_size(self.config.block_size)
            self.hyperparams["block_size"] = self.config.block_size # so that the checkpoint will have the right value

    def prepare_data(self, encoded_data, data_division=1):
        """
        1. `encoded_data`: The encoded training text data.

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
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
        x = torch.stack([data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.config.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self, eval_iters):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y)

                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.config.warmup_iters:
            return self.config.learning_rate * it / self.config.warmup_iters

        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config.lr_decay_iters:
            return self.config.min_lr

        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)

        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def train(self, max_iters, eval_interval, log_interval, eval_iters):
        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=False)

        # compile the model
        if self.config.compile:
            print(f"Compiling the model... {Fore.BLACK}{Style.BRIGHT}(takes a ~minute)")
            self.model = torch.compile(self.model) # requires PyTorch 2.0

        # training loop
        X, Y = self.get_batch("train") # fetch the very first batch
        t0 = time.time()
        t1 = time.time()
        local_iter_num = 0 # number of iterations in the lifetime of this process
        running_mfu = -1.0
        loop = 1
        while loop == 1:
            # determine and set the learning rate for this iteration
            lr = self.get_lr(self.iter_num) if self.config.decay_lr else self.config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if self.iter_num % eval_interval == 0:
                losses = self.estimate_loss(eval_iters)

                print(
                    f"{Fore.WHITE}{Style.BRIGHT}step "
                    f"{Fore.BLACK}{Style.BRIGHT}[{self.iter_num}/{max_iters}]"
                    f"{Fore.RESET}{Style.RESET_ALL}: "
                    f"train loss {Fore.WHITE}{Style.BRIGHT}{losses["train"]:.4f}"
                    f"{Fore.RESET}{Style.RESET_ALL}, "
                    f"val loss {Fore.WHITE}{Style.BRIGHT}{losses["val"]:.4f}"
                    f"{Fore.RESET}{Style.RESET_ALL}, "
                    f"lr {Fore.WHITE}{Style.BRIGHT}{lr}"
                    f"{Fore.RESET}{Style.RESET_ALL}, "
                    f"mfu {Fore.WHITE}{Style.BRIGHT}{running_mfu*100}"
                    f"time took {Fore.BLACK}{Style.BRIGHT}{calc_total_time(time.time() - t1)}"
                )

                self.losses["train"].append(losses["train"])
                self.losses["val"].append(losses["val"])

            if self.config.checkpoints and (iter + 1) % self.config.checkpoints["interval"] == 0:
                if not os.path.isdir(self.config.checkpoints["path"]):
                    os.mkdir(self.config.checkpoints["path"])

                if self.iter_num > 0:
                    torch.save(self.get_trained_model(), f"{self.config.checkpoints["path"]}\\{self.config.checkpoints["name"]}_step{(iter + 1)}.pt")

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(self.config.gradient_accumulation_steps):
                with self.ctx:
                    logits, loss = self.model(X, Y)
                    loss = loss / self.config.gradient_accumulation_steps # scale the loss to account for gradient accumulation

                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = self.get_batch("train")
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()

            # clip the gradient
            if self.config.grad_clip != 0.0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # step the optimizer and scaler if training in fp16
            scaler.step(self.optimizer)
            scaler.update()

            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if self.iter_num % log_interval == 0:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * self.config.gradient_accumulation_steps

                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = self.model.estimate_mfu(self.config.batch_size * self.config.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu

                print(f"iter {self.iter_num}: loss {lossf:.4f}, time {calc_total_time(dt)}ms, mfu {running_mfu*100:.2f}%")

            self.iter_num += 1
            local_iter_num += 1

            # termination conditions
            if self.iter_num > max_iters:
                break

        return self.get_trained_model()

    def get_trained_model(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "hyperparams": self.hyperparams,
            "iter_num": self.iter_num,
            "device": self.device
        }
