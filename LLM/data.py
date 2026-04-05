# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def _sample_from_token_buffer(token_buffer, nsamples, seqlen, seed):
    total_tokens = token_buffer.shape[1]
    if total_tokens < seqlen:
        raise ValueError(
            f"Not enough tokens to build a sequence of length {seqlen}. "
            f"Only found {total_tokens} tokens."
        )

    random.seed(seed)
    trainloader = []
    max_start = total_tokens - seqlen
    for _ in range(nsamples):
        start = 0 if max_start == 0 else random.randint(0, max_start)
        inp = token_buffer[:, start:start + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def _build_token_buffer_from_texts(texts, tokenizer, seqlen):
    chunks = []
    total_tokens = 0

    for text in texts:
        if not text:
            continue

        enc = tokenizer(text, return_tensors='pt')
        input_ids = enc.input_ids
        if input_ids.numel() == 0:
            continue

        chunks.append(input_ids)
        total_tokens += input_ids.shape[1]

        if total_tokens >= seqlen:
            break

    if not chunks:
        raise ValueError("Dataset did not contain any tokenizable text.")

    return torch.cat(chunks, dim=1)

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    trainloader = _sample_from_token_buffer(trainenc.input_ids, nsamples, seqlen, seed)
    return trainloader, testenc


# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    random.seed(seed)
    
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
    # valdata = load_dataset(
    #     'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )
    
    trainenc = _build_token_buffer_from_texts(
        (sample["text"] for sample in traindata), tokenizer, seqlen
    )
    trainloader = _sample_from_token_buffer(trainenc, nsamples, seqlen, seed)

    # Prepare validation dataset
    valenc = None
    # valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    # valenc = valenc.input_ids[:, :(256 * seqlen)]
    # valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
