from ...utils import one_hot_encoding, remove_special_chars, tokenize
from ...models.FeedForward import FeedForwardConfig, FeedForward
from ...models.RNN import RNNConfig, RNN
import torch

class Sample:
    def __init__(self, model_path):
        # Load the saved model
        model_data = torch.load(model_path)

        self.state_dict = model_data["state_dict"]
        self.vocab = model_data["vocab"]
        self.model_architecture = model_data["model"]
        self.classes = model_data["classes"]
        self.device = model_data["device"]
        self.n_hidden = model_data["config"]["n_hidden"]
        self.n_layer = model_data["config"]["n_layer"]

    def load(self):
        # set hyperparameters
        if self.model_architecture == "FeedForward":
            FeedForwardConfig.n_layer = self.n_layer
            FeedForwardConfig.n_hidden = self.n_hidden
            FeedForwardConfig.input_size = len(self.vocab)
            FeedForwardConfig.output_size = len(self.classes)

            # create an instance of FeedForward network
            self.model = FeedForward()

        elif self.model_architecture == "RNN":
            RNNConfig.n_layer = self.n_layer
            RNNConfig.n_hidden = self.n_hidden
            RNNConfig.input_size = len(self.vocab)
            RNNConfig.output_size = len(self.classes)

            # create an instance of RNN network
            self.model = RNN()

        else:
            raise Exception(f"{self.model_architecture}: Invalid model architecture.\nAvailable architectures are FeedForward, RNN")

        # load the saved model state_dict
        self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()  # set the model to evaluation mode

    # use the model for classification or other tasks
    def predict(self, text):
        sentence = remove_special_chars(tokenize(text))

        X = one_hot_encoding(sentence, self.vocab)
        X = X.reshape(1, X.shape[0])
        X = torch.tensor(X).to(self.device)

        return self.model.predict(X, self.classes)
