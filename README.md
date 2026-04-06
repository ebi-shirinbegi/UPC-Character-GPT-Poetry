# Character-Level GPT for Persian Poetry

A decoder-only Transformer trained on classical Persian poetry (ghazals from the [Ganjoor](https://huggingface.co/datasets/kakooch/ganjoor-processed) dataset) to generate verse and study how poetic structure emerges in neural representations.

Built for **Lab 3** of the *Theory and Practice of Large Language Models* course at Université Paris Cité, taught by Professor Guillaume Wisniewski.

## What This Does

- Implements a character-level GPT from scratch using PyTorch
- Trains on ~6,300 classical Persian poems (8.4 MB, 96-char vocabulary)
- Compares FlashAttention vs masked attention (speed and memory)
- Explores hyperparameters: number of layers (2/4/6) and context length (128/256/512)
- Generates poems and statistically analyzes them against the training corpus (line lengths, character frequencies, rhyme consistency, stanza structure)

## Model

| Parameter | Value |
|-----------|-------|
| Architecture | Decoder-only Transformer (GPT) |
| Positional embeddings | Learned |
| Layers | 6 |
| Hidden dimension | 256 |
| Attention heads | 8 |
| Context length | 256 |
| Parameters | 4.8M |
| Final BPC | 2.2 |

## Results

The model learns to produce text with correct Persian characters, proper word boundaries, regular hemistich lengths (~28 chars), and recognizable poetic vocabulary. It partially learns rhyme patterns (radif), though consistent rhyming across full poems remains a challenge for a character-level model at this scale.

## Running

Open `lab3_notebook.ipynb` in Google Colab with a T4 GPU runtime. Run cells top to bottom. Training takes ~2 hours for the main model, ~3 hours for hyperparameter exploration.

## Author

Mohammad Ebrahim Sharifi — M1 Computational Linguistics, Université Paris Cité
