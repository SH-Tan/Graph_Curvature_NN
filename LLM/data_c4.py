import random
import torch
from datasets import load_dataset


class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_c4_independent(nsamples, seed, seqlen, tokenizer):
    random.seed(seed)

    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    trainloader = _sample_independent_examples(traindata, nsamples, seqlen, tokenizer, seed)
    valenc = None
    return trainloader, valenc


def _sample_independent_examples(dataset, nsamples, seqlen, tokenizer, seed):
    random.seed(seed)

    valid_examples = []
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for idx in indices:
        text = dataset[idx]["text"]
        if not text:
            continue

        enc = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = enc.input_ids

        if input_ids.shape[1] < seqlen:
            continue

        # choose one contiguous chunk from this single text only
        max_start = input_ids.shape[1] - seqlen
        start = 0 if max_start == 0 else random.randint(0, max_start)
        inp = input_ids[:, start:start + seqlen]

        tar = inp.clone()
        tar[:, :-1] = -100

        valid_examples.append((inp, tar))

        if len(valid_examples) == nsamples:
            break

    if len(valid_examples) < nsamples:
        raise ValueError(
            f"Could only build {len(valid_examples)} independent examples, "
            f"but nsamples={nsamples} was requested."
        )

    return valid_examples




def get_c4_dependent(nsamples, seed, seqlen, tokenizer, stride=1):
    random.seed(seed)

    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    token_buffer = _sample_overlapping_source_text(
        traindata,
        tokenizer,
        nsamples,
        seqlen,
        stride,
        seed,
    )

    trainloader = _sample_dependent_examples(token_buffer, nsamples, seqlen, stride)
    valenc = None
    return trainloader, valenc


def _sample_dependent_examples(token_buffer, nsamples, seqlen, stride=1):
    total_tokens = token_buffer.shape[1]
    needed = seqlen + (nsamples - 1) * stride

    if total_tokens < needed:
        raise ValueError(
            f"Not enough tokens. Need at least {needed}, found {total_tokens}."
        )

    trainloader = []
    for i in range(nsamples):
        start = i * stride
        inp = token_buffer[:, start:start + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def _sample_overlapping_source_text(dataset, tokenizer, nsamples, seqlen, stride, seed):
    needed = seqlen + (nsamples - 1) * stride

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    for idx in indices:
        text = dataset[idx]["text"]
        if not text:
            continue

        enc = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = enc.input_ids
        total_tokens = input_ids.shape[1]

        # Dependent samples must all come from one document. If this text is too
        # short, skip it and try the next text instead of concatenating documents.
        if total_tokens < needed:
            continue

        max_start = total_tokens - needed
        start = 0 if max_start == 0 else rng.randint(0, max_start)
        return input_ids[:, start:start + needed]

    raise ValueError(
        "Could not find a single text long enough for dependent examples. "
        f"Need at least {needed} tokens in one document."
    )


# Function to select the appropriate loader based on dataset name
def get_loaders_c4(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if "c4_independent" in name:
        return get_c4_independent(nsamples, seed, seqlen, tokenizer)
    if "c4_dependent" in name:
        return get_c4_dependent(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Unsupported dataset name: {name}")
