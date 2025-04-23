import torch as t
from transformers import GPTNeoXForCausalLM, AutoTokenizer

from .utils import CACHE_DIR
from .utils import DEVICE

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
        revision="step143000",
        # attn_implementation="sdpa",
        cache_dir=f"{CACHE_DIR}/reverse-pythia-160m",
    ).to(DEVICE)

    ba_tokenizer = AutoTokenizer.from_pretrained(
        "afterless/reverse-pythia-160m",
        revision="step143000",
        cache_dir=f"{CACHE_DIR}/reverse-pythia-160m",
    )
    #
    ba_tokenizer.pad_token = "<pad>"
    ba_tokenizer.pad_token_id = 0

    return ba_model, ba_tokenizer