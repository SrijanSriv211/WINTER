import torch

class Sample:
    def __init__(self, model_path):
        # Load the saved model
        model_data = torch.load(model_path)

    def load(self):
        pass

    # use the model for classification or other tasks
    def extract(self, text):
        pass
