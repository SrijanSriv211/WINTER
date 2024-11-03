from ..shared.utils import calc_total_time
from colorama import Style, Fore, init
from torch.nn import functional as F
from contextlib import nullcontext
from dataclasses import dataclass
import inspect, pickle, random, time, math, os
import torch.nn as nn, torch.amp, torch
import matplotlib.pyplot as plt

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

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print(f"{Fore.RED}{Style.BRIGHT}WARNING:", "using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 4282
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 32
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"Num decayed parameter tensors: {Fore.WHITE}{Style.BRIGHT}{len(decay_params)}"
            f"{Style.RESET_ALL},",
            f"with {Fore.WHITE}{Style.BRIGHT}{num_decay_params:,}",
            "parameters"
        )

        print(
            f"Num non-decayed parameter tensors: {Fore.WHITE}{Style.BRIGHT}{len(nodecay_params)}"
            f"{Style.RESET_ALL},",
            f"with {Fore.WHITE}{Style.BRIGHT}{num_nodecay_params:,}",
            "parameters"
        )

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        color = f"{Fore.LIGHTGREEN_EX}{Style.BRIGHT}" if use_fused == True else f"{Fore.LIGHTRED_EX}{Style.BRIGHT}"
        print(f"Using fused AdamW: {color}{use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

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
    def __init__(self, device="auto"):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    def load(self, checkpoint, compile=False):
        # create an instance of GPT
        gptconf = GPTConfig(**checkpoint["hyperparams"])
        self.model = GPT(gptconf)

        # remove `_orig_mod.` prefix from state_dict (if it's there)
        state_dict = checkpoint["model"]
        unwanted_prefix = '_orig_mod.'

        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        # load the saved model state_dict
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval() # set the model to evaluation mode

        if compile:
            #NOTE: backend="inductor" is giving some errors so switched to aot_eager.
            self.model = torch.compile(self.model, backend="aot_eager") # requires PyTorch 2.0

    # use the model for generation or other tasks
    def generate(self, encoded_text=None, length=100, temperature=0.7, top_k=50):
        """
        `max_new_tokens`: number of tokens generated in each sample
        `temperature`: 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        `tok_k`: retain only the top_k most likely tokens, clamp others to have 0 probability
        """

        return self.model.generate(self.prepare_context(encoded_text), max_new_tokens=length, temperature=temperature, top_k=top_k)[0].tolist()

    def prepare_context(self, encoded_text):
        if encoded_text == None:
            return torch.zeros((1, 1), dtype=torch.long, device=self.device)

        return torch.tensor(encoded_text, dtype=torch.long, device=self.device).unsqueeze(0)

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
            gradient_accumulation_steps = 1, # used to simulate larger batch sizes
            batch_size = 32, # if gradient_accumulation_steps > 1, this is the micro-batch size
            block_size = 50,

            # model
            vocab_size = 2482,
            n_layer = 4,
            n_head = 4,
            n_embd = 32,
            dropout = 0.0, # for pretraining 0 is good, for finetuning try 0.1+
            bias = False, # do we use bias inside LayerNorm and Linear layers?

            # adamw optimizer
            learning_rate = 4e-2, # max learning rate
            weight_decay = 1e-1,
            beta1 = 0.9,
            beta2 = 0.95,
            grad_clip = 1, # clip gradients at this value, or disable if == 0.0

            # learning rate decay settings
            decay_lr = True, # whether to decay the learning rate
            warmup_iters = 100, # how many steps to warm up for
            lr_decay_iters = 1000, # should be ~= max_iters per Chinchilla
            min_lr = 4e-3, # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

            # system
            device = "cpu", # examples: "cpu", "cuda", "cuda:0", "cuda:1" etc., or try "mps" on macbooks
            seed = "auto", # examples: "auto", 1337 or any other number
            compile = False # use PyTorch 2.0 to compile the model to be faster
        )
        ```
        """

        self.config = config
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if self.config["device"] == "auto" else self.config["device"]

        # init seed
        torch.manual_seed(self.config["seed"]) if self.config["seed"] != "auto" else None
        random.seed(self.config["seed"]) if self.config["seed"] != "auto" else None

        # "float32", "bfloat16", or "float16", the latter will auto implement a GradScaler
        dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        self.ctx = nullcontext() if self.device == "cpu" else torch.amp.autocast(device_type=self.device, dtype=ptdtype)

        # print the device
        print("Training on", f"{Fore.YELLOW}{Style.BRIGHT}{self.device}", f"{Fore.WHITE}{Style.BRIGHT}({torch.initial_seed()})")

    def from_scratch(self):
        self.hyperparams = dict(dropout=self.config["dropout"])
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            self.hyperparams[k] = self.config[k]

        gptconf = GPTConfig(**self.hyperparams)
        # create an instance of GPT
        self.model = GPT(gptconf)
        self.model.to(self.device)

        # optimizer
        self.optimizer = self.model.configure_optimizers(self.config["weight_decay"], self.config["learning_rate"], (self.config["beta1"], self.config["beta2"]), self.config["device"])

        # a dict for keep track of all the losses to be plotted.
        self.metrics = {
            "train": [],
            "eval": [],
            "val": [],
            "mfu": [],
            "lr": []
        }
        self.iter_num = 0
        self.best_loss = 0

    def from_pretrained(self, checkpoint: dict):
        """
        Load the pretrained model from path, for eg, `from_pretrained(torch.load("bin\\model\\ckpt.pth", map_location=self.device))`
        """

        # make loading pretrained models backwards compatible with previously trained models
        metrics = [checkpoint[i] for i in ["metrics", "losses"] if i in checkpoint.keys()]
        self.metrics = {
            "train": [],
            "eval": [],
            "val": [],
            "mfu": [],
            "lr": []
        } if not metrics else metrics[0]
        for i in ["mfu", "lr"]:
            if i not in self.metrics.keys():
                self.metrics[i] = []

        # load the state dict and current iteration number of the model
        state_dict = checkpoint["model"]
        self.iter_num = checkpoint["iter_num"]
        self.best_loss = checkpoint["best_loss"] if "best_loss" in checkpoint.keys() else 0

        self.hyperparams = dict(dropout=self.config["dropout"])
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            self.hyperparams[k] = checkpoint["hyperparams"][k]

        gptconf = GPTConfig(**self.hyperparams)

        # create an instance of GPT
        self.model = GPT(gptconf)

        # remove `_orig_mod.` prefix from state_dict (if it's there)
        state_dict = checkpoint["model"]
        unwanted_prefix = '_orig_mod.'

        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        # optimizer
        self.optimizer = self.model.configure_optimizers(self.config["weight_decay"], self.config["learning_rate"], (self.config["beta1"], self.config["beta2"]), self.config["device"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # crop down the model block size if desired, using model surgery
        if self.config["block_size"] < self.hyperparams["block_size"]:
            self.model.crop_block_size(self.config["block_size"])
            self.hyperparams["block_size"] = self.config["block_size"] # so that the checkpoint will have the right value

    def prepare_data(self, encoded_data, path="bin", data_division=1):
        """
        Generate `train.bin` and `val.bin` from encoded data.

        Use this only once when you don't have `train.bin` and `val.bin`

        If you already have `train.bin` and `val.bin`, then use `get_data` function.

        1. `encoded_data`: The encoded training text data.

        For eg,
        ```python
        encode(text, stoi=self.stoi)
        ```

        2. `data_division`: The first `(data_division * 100)%` will be train, rest val
        3. `path`: Path where `train.bin` and `val.bin` will be saved
        """

        data = torch.tensor(encoded_data, dtype=torch.long)

        # train and test splits
        n = int(data_division * len(data)) # the first (data_division * 100)% will be train, rest val
        train_data = data[:n]
        val_data = data[n:] if 0 < data_division < 1 else data[:n]

        self.train_data = f"{path}\\train.bin"
        self.val_data = f"{path}\\val.bin"

        with open(self.train_data, "wb") as f:
            pickle.dump(train_data, f)

        with open(self.val_data, "wb") as f:
            pickle.dump(val_data, f)

        # print the number of tokens
        print(f"{Fore.WHITE}{Style.BRIGHT}{(len(data)/1e6)}M", "total tokens")
        print(
            f"{Fore.WHITE}{Style.BRIGHT}{(len(train_data)/1e6)}M", "train tokens,",
            f"{Fore.WHITE}{Style.BRIGHT}{(len(val_data)/1e6)}M", "test tokens",
            f"   {Fore.WHITE}{Style.DIM}(Using train tokens as test tokens)" if not (0 < data_division < 1) else ""
        )

    def get_data(self, train, val, is_file=True):
        """
        1. `train`: Path to training data (`train.bin`)
        2. `val`: Path to validation data (`val.bin`)
        """

        self.is_dataset_a_file = is_file
        self.train_data = train
        self.val_data = val

        train_data, val_data = 0, 0
        if is_file:
            # Try to load and check all the data
            with open(self.train_data, "rb") as f:
                train_data = len(pickle.load(f))

            with open(self.val_data, "rb") as f:
                val_data = len(pickle.load(f))

        else:
            for i in os.listdir(self.train_data):
                # Try to load and check all the data
                with open(f"{self.train_data}\\{i}", "rb") as f:
                    train_data += len(pickle.load(f))

            for i in os.listdir(self.val_data):
                with open(f"{self.val_data}\\{i}", "rb") as f:
                    val_data += len(pickle.load(f))

        data = train_data + val_data

        # print the number of tokens
        print(f"{Fore.WHITE}{Style.BRIGHT}{(data/1e6)}M", "total tokens")
        print(
            f"{Fore.WHITE}{Style.BRIGHT}{(train_data/1e6)}M", "train tokens,",
            f"{Fore.WHITE}{Style.BRIGHT}{(val_data/1e6)}M", "test tokens",
            f"   {Fore.WHITE}{Style.DIM}(Using train tokens as test tokens)" if not (0 < (train_data/data) < 1) else ""
        )

    def _load_data(self, path):
        if not self.is_dataset_a_file:
            files = os.listdir(path)
            random.shuffle(files)

        with open(f"{path}\\{files[0]}" if not self.is_dataset_a_file else path, "rb") as f:
            return pickle.load(f)

    # data loading
    # generate a small batch of data of inputs x and targets y
    def get_batch(self, split):
        # We reload data every batch to avoid a memory leak
        path = self.train_data if split == "train" else self.val_data
        data = self._load_data(path)

        ix = torch.randint(len(data) - self.config["block_size"], (self.config["batch_size"],))
        x = torch.stack([data[i:i+self.config["block_size"]] for i in ix])
        y = torch.stack([data[i+1:i+self.config["block_size"]+1] for i in ix])
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
        if it < self.config["warmup_iters"]:
            return self.config["learning_rate"] * it / self.config["warmup_iters"]

        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config["lr_decay_iters"]:
            return self.config["min_lr"]

        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.config["warmup_iters"]) / (self.config["lr_decay_iters"] - self.config["warmup_iters"])

        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.config["min_lr"] + coeff * (self.config["learning_rate"] - self.config["min_lr"])

    def train(self, max_iters=1000, eval_interval=100, log_interval=100, eval_iters=100, patience=10):
        # report number of parameters
        print(f"{Fore.WHITE}{Style.BRIGHT}{self.model.get_num_params()/1e6}M", "parameters")

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.amp.GradScaler(enabled=False)

        # compile the model
        if self.config["compile"]:
            print(f"Compiling the model... {Fore.BLACK}{Style.BRIGHT}(takes a ~minute)")
            #NOTE: backend="inductor" is giving some errors so switched to aot_eager.
            self.model = torch.compile(self.model, backend="aot_eager") # requires PyTorch 2.0

        # training loop
        X, Y = self.get_batch("train") # fetch the very first batch
        start_time = time.time()
        eval_t0 = time.time()
        t0 = time.time()
        local_iter_num = 0 # number of iterations in the lifetime of this process
        running_mfu = -1.0
        interval_without_improvement = 0

        while True:
            try:
                # determine and set the learning rate for this iteration
                lr = self.get_lr(self.iter_num) if self.config["decay_lr"] else self.config["learning_rate"]
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                # evaluate the loss on train/val sets and write checkpoints
                if self.iter_num % eval_interval == 0:
                    losses = self.estimate_loss(eval_iters)
                    # timing and logging
                    eval_t1 = time.time()
                    eval_dt = eval_t1 - eval_t0
                    eval_t0 = eval_t1

                    print(
                        f"{Fore.WHITE}{Style.BRIGHT}step",
                        f"{Fore.BLACK}{Style.BRIGHT}[{self.iter_num}/{max_iters}]"
                        f"{Fore.RESET}{Style.RESET_ALL}:",
                        f"train loss {Fore.WHITE}{Style.BRIGHT}{losses["train"]:.4f}"
                        f"{Fore.RESET}{Style.RESET_ALL},",
                        f"val loss {Fore.WHITE}{Style.BRIGHT}{losses["val"]:.4f}"
                        f"{Fore.RESET}{Style.RESET_ALL},",
                        f"lr {Fore.WHITE}{Style.BRIGHT}{lr:.4f}"
                        f"{Fore.RESET}{Style.RESET_ALL},",
                        f"time took {Fore.BLACK}{Style.BRIGHT}{calc_total_time(eval_dt)}"
                    )

                    self.metrics["train"].append(losses["train"])
                    self.metrics["val"].append(losses["val"])

                    # check for early stopping
                    if losses["val"] < self.best_loss:
                        self.best_loss = losses["val"]
                        interval_without_improvement = 0  # reset the count

                    elif interval_without_improvement + 1 >= patience:
                        print("Early stopping due to no improvement in validation loss.")
                        break

                    else:
                        interval_without_improvement += 1

                # save checkpoint
                if self.config["checkpoints"] and self.iter_num % self.config["checkpoints"]["interval"] == 0:
                    if not os.path.isdir(self.config["checkpoints"]["path"]):
                        os.mkdir(self.config["checkpoints"]["path"])

                    if self.iter_num > 0:
                        print(f"saved checkpoint at step {Fore.WHITE}{Style.BRIGHT}{self.iter_num}")
                        torch.save(self.get_trained_model(), f"{self.config["checkpoints"]["path"]}\\{self.config["checkpoints"]["name"]}_step{self.iter_num}.pth")

                # forward backward update, with optional gradient accumulation to simulate larger batch size
                # and using the GradScaler if data type is float16
                for micro_step in range(self.config["gradient_accumulation_steps"]):
                    with self.ctx:
                        logits, loss = self.model(X, Y)
                        loss = loss / self.config["gradient_accumulation_steps"] # scale the loss to account for gradient accumulation

                    # immediately async prefetch next batch while model is doing the forward pass on the GPU
                    X, Y = self.get_batch("train")
                    # backward pass, with gradient scaling if training in fp16
                    scaler.scale(loss).backward()

                # clip the gradient
                if self.config["grad_clip"] != 0.0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])

                # step the optimizer and scaler if training in fp16
                scaler.step(self.optimizer)
                scaler.update()

                # flush the gradients as soon as we can, no need for this memory anymore
                self.optimizer.zero_grad(set_to_none=True)

                print(
                    f"{Fore.WHITE}{Style.BRIGHT}iter",
                    f"{Fore.BLACK}{Style.BRIGHT}[{self.iter_num}/{max_iters}]"
                    f"{Fore.RESET}{Style.RESET_ALL}:",
                    f"loss {Fore.WHITE}{Style.BRIGHT}{(loss.item() * self.config["gradient_accumulation_steps"]):.4f}",
                    end="\r"
                )

                # timing and logging
                if self.iter_num % log_interval == 0:
                    t1 = time.time()
                    dt = t1 - t0
                    t0 = t1

                    # get loss as float. note: this is a CPU-GPU sync point
                    # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                    lossf = loss.item() * self.config["gradient_accumulation_steps"]

                    if local_iter_num >= 5: # let the training loop settle a bit
                        mfu = self.model.estimate_mfu(self.config["batch_size"] * self.config["gradient_accumulation_steps"], dt)
                        running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu

                    print(
                        f"{Fore.WHITE}{Style.BRIGHT}iter",
                        f"{Fore.BLACK}{Style.BRIGHT}[{self.iter_num}/{max_iters}]"
                        f"{Fore.RESET}{Style.RESET_ALL}:",
                        f"loss {Fore.WHITE}{Style.BRIGHT}{lossf:.4f}"
                        f"{Fore.RESET}{Style.RESET_ALL},",
                        f"mfu {Fore.WHITE}{Style.BRIGHT}{running_mfu*100:.2f}"
                        f"{Fore.RESET}{Style.RESET_ALL},",
                        f"time took {Fore.BLACK}{Style.BRIGHT}{calc_total_time(dt)}"
                    )
                    self.metrics["mfu"].append(running_mfu)
                    self.metrics["eval"].append(lossf)
                    self.metrics["lr"].append(lr)

                self.iter_num += 1
                local_iter_num += 1

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
            "optimizer": self.optimizer.state_dict(),
            "hyperparams": self.hyperparams,
            "iter_num": self.iter_num,
            "device": self.device,
            "metrics": self.metrics,
            "best_loss": self.best_loss
        }

    # def plot(self, path):
    #     self._plot("train-val loss", [(self.metrics["train"], "train loss"), (self.metrics["val"], "val loss")], path + "-train-val.png")
    #     self._plot("eval loss", [(self.metrics["eval"], "eval loss")], path + "-eval.png")
    #     self._plot("mfu", [(self.metrics["mfu"], "mfu")], path + "-mfu.png")
    #     self._plot("lr", [(self.metrics["lr"], "lr")], path + "-lr.png")

    # def _plot(self, title, plot_data, save_path):
    #     with plt.style.context("seaborn-v0_8-dark"):
    #         for param in ["figure.facecolor", "axes.facecolor", "savefig.facecolor"]:
    #             plt.rcParams[param] = "#030407"

    #         for param in ["text.color", "axes.labelcolor", "xtick.color", "ytick.color"]:
    #             plt.rcParams[param] = "0.9"

    #         plt.figure(figsize=(18, 8))

    #         for losses, label in plot_data:
    #             plt.plot(losses, label=label)

    #         plt.xlabel("iteration", fontsize=12)
    #         plt.ylabel("value", fontsize=12)
    #         plt.legend(fontsize=12)
    #         plt.title(title, fontsize=14)
    #         plt.savefig(save_path, bbox_inches="tight")
    #         plt.close()
