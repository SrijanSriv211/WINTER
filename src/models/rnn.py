from colorama import Style, Fore, init
from dataclasses import dataclass
from torch.nn import functional as F
import torch.nn as nn, torch.amp, torch, inspect

init(autoreset=True)

@dataclass
class RNNConfig:
    input_size: int = None
    output_size: int = None
    n_hidden: int = 4
    n_layer: int = 4
    dropout: int = 0

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        assert config.input_size is not None
        assert config.output_size is not None
        self.config = config

        # Defining the layers
        self.rnn = nn.RNN(config.input_size, config.n_hidden, config.n_layer, batch_first=True, dropout=config.dropout) # RNN Layer
        self.fc = nn.Linear(config.n_hidden, config.output_size) # Fully connected layer

    def forward(self, x, targets=None):
        # Apply RNN layer
        out, _ = self.rnn(x)

        # Squeeze the dimensions to remove the batch_first dimension
        out = self.fc(out)

        # Calculate loss
        loss = None if targets is None else F.cross_entropy(out, targets)
        return out, loss

    def configure_optimizers(self, learning_rate, device_type):
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        color = f"{Fore.LIGHTGREEN_EX}{Style.BRIGHT}" if use_fused == True else f"{Fore.LIGHTRED_EX}{Style.BRIGHT}"
        print(f"Using fused AdamW: {color}{use_fused}")
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, betas=(0.9, 0.95), fused=use_fused)
        return optimizer

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
    def __init__(self, device="auto"):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    def load(self, checkpoint, compile=False):
        # create an instance of RNN
        rnnconf = RNNConfig(**checkpoint["hyperparams"])
        self.model = RNN(rnnconf)

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

    # use the model for classification or other tasks
    def predict(self, X, classes):
        i, confidence = self.model.predict(self.prepare_context(X))
        tag = classes[i]

        return tag, confidence

    def prepare_context(self, X):
        X = X.reshape(1, X.shape[0])
        return torch.tensor(X).to(self.device)
