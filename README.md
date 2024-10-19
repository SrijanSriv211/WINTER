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
