from .model import GPTConfig, GPT
import torch

class sample:
    def __init__(self, checkpoint, device="auto", auto_load=True):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

        self.state_dict = checkpoint["model"]
        self.gptconf = GPTConfig(**checkpoint["hyperparams"])

        # automatically load the model
        if auto_load: self.load()

    def load(self):
        # create an instance of RNN
        self.model = GPT(self.gptconf)

        # load the saved model state_dict
        self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()  # set the model to evaluation mode

    # use the model for generation or other tasks
    def generate(self, encoded_text=None, length=100, temperature=1.0, top_k=None):
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
