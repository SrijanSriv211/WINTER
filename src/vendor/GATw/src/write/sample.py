from ..models.GPT import GPTConfig, GPT
from ..utils import encode, decode
import torch

class Sample:
    def __init__(self, model_path):
        # Load the saved model
        model_data = torch.load(model_path)

        self.state_dict = model_data["state_dict"]
        self.stoi = model_data["stoi"]
        self.itos = model_data["itos"]
        self.device = model_data["device"]
        self.n_embd = model_data["config"]["n_embd"]
        self.n_head = model_data["config"]["n_head"]
        self.n_layer = model_data["config"]["n_layer"]
        self.block_size = model_data["config"]["block_size"]
        self.dropout = model_data["config"]["dropout"]
        self.vocab_size = model_data["config"]["vocab_size"]

    def load(self):
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

        # Load the saved model state_dict
        self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

    # Use the model for generation or other tasks
    def generate(self, text="", length=100, temperature=1.0, top_k=None):
        """
        @param max_new_tokens: number of tokens generated in each sample
        @param temperature: 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        @param tok_k: retain only the top_k most likely tokens, clamp others to have 0 probability
        """

        if text == "":
            context = torch.zeros((1, 1), dtype=torch.long, device=self.device)

        else:
            context = torch.tensor(encode(text, stoi=self.stoi), dtype=torch.long, device=self.device).unsqueeze(0)

        return decode(self.model.generate(context, max_new_tokens=length, temperature=temperature, top_k=top_k)[0].tolist(), itos=self.itos)
