# WINTER
***A breakthrough in Intelligence with Language Applications***

**WINTER** is an **open-source personal assistant** who lives **in your computer**.

WINTER Stands for **Witty Intelligence with Natural Emotions and Rationality**

He **does stuff** when you **ask him to**.

You can **talk to him** and he can **talk to you**.
You can also **text him** and he can also **text you**.
If you want to, WINTER can communicate with you by being **offline to protect your privacy**.

Moreover as his name suggests, WINTER is very **smart with short, quick witted responses** with a **subtle pinch of human emotions** embedded into him via his training.

### Why WINTER?
> 1. With WINTER you can do a lot of your **daily tasks automatically** making your life a bit easier.
> 2. You to **teach him new skills to expand** his capabilities to **help you work on great projects & products**.
> 4. Since privacy is an important factor, WINTER for most part works completely offline. To go 100% offline you can text him.
> 3. WINTER is cool because it is AI.
> 5. Open source is great.

### What can WINTER do for me?
> WINTER is highly scalable, thus he can do a lot of tasks for you.
> From the basics like controlling display brightness to more advanced stuff like working together on an advanced C++ project.
> Basically WINTER can do anything you want.

---

## :toolbox: Getting Started
### :bangbang: Prerequisites
You need to install the following on your machine.
- [Python](https://www.python.org/) >= 3.12

### :pencil: Getting Started
<ins>**1. Downloading the repository:**</ins>

Start by cloning the repository with `git clone --recursive https://github.com/SrijanSriv211/WINTER`.

If the repository was cloned non-recursively previously, use `git submodule update --init --recursive` to clone the necessary submodules.

<ins>**2. Setting up virtual environment:**</ins>

```console
> python -m venv .env
> .env\Scripts\activate
(.env)> pip install requirements.txt
```

### Naming convention for trained models
> 1. **cl** stands for **core language**
>    For eg, **cl100k** = **core language 100k**
>    ∴ Text tokenizer with 100k tokens
> 
> 2. **is** stands for **intentions**
>    For eg, **is10** = **intentions with 10 classes**
>    ∴ Classification model with 10 classes
> 
> 3. **aw** stands for **alphabet write**
>    For eg, **aw100k** = **alphabet write 100k parameters**
>    ∴ Generative model with 100k parameters
> 
> 4. **ae** stands for **auto encoder**
>    For eg, **ae100k** = **auto encoder 100k parameters**
>    ∴ Auto encoder-decoder model with 100k parameters

> For example:
> 1. `cl100kaw100k` is a **text generation model with 100k parameters trained on 100k tokens**
> 2. `cl1kis10` is a **text classification model with 10 classes trained on 1k tokens**

<ins>**3. Train encoder:**</ins>

```python
from src.models.encoder import Encoder
enc = Encoder()
enc.train("dataset.txt", 8279, batch_size=100, drop_bounds_after=4279, is_file=True)
enc.register_special_tokens({"<|pad|>": 8279, "<|sot|>": 8280, "<|eot|>": 8281})
enc.save("bin\\cl8k.bin")
```

<ins>**4. Train WINTER:**</ins>

First create a `json` file in scripts folder called `config.json` with the following format

```json
{
	"GPT": {
		"load_from_file": false,
		"train_data": "data\\GATw\\train",
		"val_data": "data\\GATw\\val",
		"init_from": "pretrained,bin\\GATw.bin",
		// "init_from": "scratch",

		"checkpoints": {
			"path": "bin\\ck",
			"name": "GATw",
			"interval": 200
		},
		"save_path": "bin\\GATw.bin",

		"max_iters": 2000,
		"eval_interval": 200,
		"log_interval": 10,
		"eval_iters": 1000,

		"gradient_accumulation_steps": 8,
		"batch_size": 16,
		"block_size": 256,

		"vocab_size": 8282,
		"n_layer": 8,
		"n_head": 4,
		"n_embd": 512,
		"dropout": 0,

		"learning_rate": 1e-3,
		"weight_decay": 0,
		"grad_clip": 1,

		"decay_lr": true,
		"warmup_iters": 500,
		"lr_decay_iters": 2000,
		"min_lr": 1e-4,

		"device": "cpu",
		"seed": "auto",
		"compile": true
	},

	"RNN": {
		"load_from_file": true,
		"train_data": "data\\clis\\train.bin",
		"val_data": "data\\clis\\train.bin",

		"checkpoints": null,
		"save_path": "bin\\clis.bin",

		"max_iters": 5000,
		"eval_interval": 500,
		"log_interval": 100,
		"eval_iters": 100,

		"batch_size": 32,
		"input_size": 4282,
		"output_size": 19,
		"n_layer": 2,
		"n_hidden": 8,
		"dropout": 0.0,

		"learning_rate": 1e-3,

		"device": "cpu",
		"seed": "auto",
		"compile": true
	}
}
```

Then tweak the values as you like. After that run the following command in the terminal

1. If you want to train the GPT model
```console
> python prepare_gpt.py
> python train_gpt.py
> python test_gpt.py
```

2. If you want to train the RNN model
```console
> python prepare_rnn.py
> python train_rnn.py
> python test_rnn.py
```
