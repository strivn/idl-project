import torch as t
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from .utils import CACHE_DIR, DEVICE

print(f"Using device: {DEVICE}")

def load_fo_model():
    # Forward model
    fo_model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m-deduped",
        revision="step143000",
        # attn_implementation="sdpa",
        cache_dir=f"{CACHE_DIR}/pythia-160m-deduped/step143000",
    ).to(DEVICE)

    fo_tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-160m-deduped",
        revision="step143000",
        cache_dir=f"{CACHE_DIR}/pythia-160m-deduped/step143000",
    )
    # tell the tokenizer what its pad token is, and its ID:
    fo_tokenizer.pad_token = "<pad>"
    fo_tokenizer.pad_token_id = 0

    return fo_model, fo_tokenizer


def load_ba_model():
    # Backward model
    ba_model = GPTNeoXForCausalLM.from_pretrained(
        "afterless/reverse-pythia-160m",
        #revision="step143000",
        # attn_implementation="sdpa",
        cache_dir=f"{CACHE_DIR}/reverse-pythia-160m",
    ).to(DEVICE)

    ba_tokenizer = AutoTokenizer.from_pretrained(
        "afterless/reverse-pythia-160m",
        #revision="step143000",
        cache_dir=f"{CACHE_DIR}/reverse-pythia-160m",
    )
    #
    ba_tokenizer.pad_token = "<pad>"
    ba_tokenizer.pad_token_id = 0

    return ba_model, ba_tokenizer


_default_ftfo_path = "/ocean/projects/cis250068p/shared/model/full-tuning-config4-restart-checkpoint-90000"
_default_ftba_path = "/ocean/projects/cis250068p/shared/model/full-tuning-reverse-config5"


def load_ftfo_model(path=_default_ftfo_path):
    # Forward model
    fo_model = GPTNeoXForCausalLM.from_pretrained(
        path,
        local_files_only=True
    ).to(DEVICE)

    fo_tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True
    )
    # tell the tokenizer what its pad token is, and its ID:
    fo_tokenizer.pad_token = "<pad>"
    fo_tokenizer.pad_token_id = 0

    return fo_model, fo_tokenizer


def load_ftba_model(path=_default_ftba_path):
    # Backward model
    ba_model = GPTNeoXForCausalLM.from_pretrained(
        path,
        local_files_only=True
    ).to(DEVICE)

    ba_tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True
    )
    
    ba_tokenizer.pad_token = "<pad>"
    ba_tokenizer.pad_token_id = 0

    return ba_model, ba_tokenizer
