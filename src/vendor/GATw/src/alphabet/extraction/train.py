import matplotlib.pyplot as plt
import torch

class Train:
    def __init__(self, n_layer, n_hidden, lr, batch_size, model="FeedForward", device="auto"):
        """
        @param n_layer: Number of layers
        @param n_hidden: Hidden size
        @param lr: Learning rate
        @param model: Model architecture to train on. [FeedForward, RNN] (default: FeedForward)
        @param device: Training device. [auto, cpu, cuda] (default: auto)
        """
        # hyperparameters
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.device = device
        self.learning_rate = lr
        self.batch_size = batch_size # how many independent sequences will we process in parallel?
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model_architecture = model

        # a dict for keep track of all the losses to be plotted.
        self.losses = {
            "train": [],
            "val": []
        }

        # print the device
        print("Training on", self.device)

    def preprocess(self, filepath, metadata, data_division:float=0.8, data_augmentation:float=0):
        """
        @param filepath: the location of the json file.
        @param metadata: (classname, tagname, pattern_name)
        @param data_division (float): if None then only train otherwise train and test (between: 0 and 1) (default: 0.8)
        @param data_augmentation (float): Duplicate a certain percentage of the original dataset during runtime (between: 0 and 1) (default: 0)
        """
        pass

    def train(self, n_steps, eval_interval, eval_iters, checkpoint_interval=0, checkpoint_path="", n_loss_digits=4):
        """
        @param n_steps: number of Epochs to train the model for
        @param eval_interval: the interval between each loss evaluation
        @param eval_iters: the iterations for each loss evaluation
        @param checkpoint_interval: the interval between each checkpoint save (default: 0)
        @param checkpoint_path: the save path for the checkpoint (default: empty string)
        @param n_loss_digits: Number of digits of train and val loss printed (default: 4)
        """

    def save(self, savepath):
        pass

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
