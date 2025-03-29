import torch as t
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# Load models
if t.cuda.is_available():
    DEVICE = "cuda"
elif t.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")
def load_fo_model():
    # Forward model
    fo_model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m-deduped",
        revision="step143000",
        cache_dir="./.cache/pythia-160m-deduped/step143000",
    ).to(DEVICE)

    fo_tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-160m-deduped",
        revision="step143000",
        cache_dir="./.cache/pythia-160m-deduped/step143000",
    )

    return fo_model, fo_tokenizer

def load_ba_model():
    # Backward model
    ba_model = GPTNeoXForCausalLM.from_pretrained(
        "afterless/reverse-pythia-160m",
        cache_dir="./.cache/reverse-pythia-160m",
    ).to(DEVICE)

    ba_tokenizer = AutoTokenizer.from_pretrained(
        "afterless/reverse-pythia-160m",
        cache_dir="./.cache/reverse-pythia-160m",
    )   
    
    return ba_model, ba_tokenizer