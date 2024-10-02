from torch.nn import functional as F
import torch.nn as nn, torch

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

class LSTMConfig:
    n_embd = 2
    n_hidden = 2
    n_layer = 1
    block_size = 16 # what is the maximum context length for predictions?
    input_size = None
    device = None

# https://github.com/SrijanSriv211/GAT-w/blob/52fb27a0da4da516400316c016d08d9317142fb2/main.py
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        # Defining the layers
        self.embedding = nn.Embedding(LSTMConfig.input_size, LSTMConfig.n_hidden)
        self.lstm = nn.LSTM(LSTMConfig.n_hidden, LSTMConfig.n_hidden, LSTMConfig.n_layer, batch_first=True)
        self.fc = nn.Linear(LSTMConfig.n_hidden, LSTMConfig.input_size) # Fully connected layer

    def forward(self, x, targets=None):
        batch_size = x.size(0)

        # Pass input through the embedding layer
        x = self.embedding(x)

        # Initialize hidden state for the first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.lstm(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, LSTMConfig.n_hidden)
        out = self.fc(out)

        # Calculate loss
        loss = None if targets is None else F.cross_entropy(out, targets)

        return out, loss

    def init_hidden(self, batch_size):
        # Initialize both hidden state and cell state
        hidden = (torch.zeros(LSTMConfig.n_layer, batch_size, LSTMConfig.n_hidden).to(LSTMConfig.device), torch.zeros(LSTMConfig.n_layer, batch_size, LSTMConfig.n_hidden).to(LSTMConfig.device))
        return hidden

    def predict(self, x, temperature=1.0):
        out, _ = self(x)

        # Adjust the output probabilities with temperature
        prob = nn.functional.softmax(out[-1] / temperature, dim=0).data

        # Sample from the modified distribution
        idx = torch.multinomial(prob, 1).item()

        return idx
