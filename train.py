from src.vendor.GATw import alphabet

t = alphabet.classification.Train(
    n_layer = 1,
    n_hidden = 8,
    lr = 4e-3,
    batch_size = 64,
    model = "RNN"
)
t.preprocess("data\\skills.json", metadata=("skills", "skill", "patterns"), data_division=None)
t.train(
    n_steps = 4000,
    eval_interval = 1000,
    eval_iters = 800,
    n_loss_digits = 7
)
t.save("bin\\skills.pth")
