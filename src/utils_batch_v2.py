import os
import torch as t
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch import amp
import math

# ----------------------
# Utils
# ----------------------

def ensure_directory_exists(directory_path):
    """
    Check if a directory exists, and create it if it doesn't.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print(f"Created directory: {directory_path}")
    return directory_path

# ----------------------
# Configs
# ----------------------

if t.cuda.is_available():
    DEVICE = "cuda"
elif t.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

if os.path.exists("/ocean"):
    CACHE_DIR = "/ocean/projects/cis250068p/shared/caches"
else:
    CACHE_DIR = ".cache/"

ensure_directory_exists(CACHE_DIR)

# PyTorch 성능 최적화 설정 추가
t.backends.cudnn.benchmark = True  # 반복적인 크기의 입력에 최적화

# ----------------------
# Scoring Function
# ----------------------

def calculate_score(
    query, answer, model, tokenizer,
    backward=False, query_direction="reverse",
    task='citation', debug=False
):
    """
    Single-pair log‑prob & perplexity.
    """
    # 1) build context & target
    if query_direction == "normal":
        assert not backward
        cond = 'is a summary of' if task == 'citation' else 'has an answer to'
        context = query + cond
        target  = answer
    else:
        if not backward:
            cond = 'is a summary of' if task == 'citation' else 'has an answer to'
            context = answer + cond
            target  = query
        else:
            cond = 'is summarized by' if task == 'citation' else 'is answered by'
            context = cond + answer
            target  = query

    # 2) tokenize
    ctx_ids = tokenizer.encode(context, return_tensors="pt")
    tgt_ids = tokenizer.encode(target,  return_tensors="pt")
    ctx_len = ctx_ids.shape[1]
    tgt_len = tgt_ids.shape[1]

    # 3) if backward, flip order
    if backward:
        ctx_ids = t.flip(ctx_ids, (1,))
        tgt_ids = t.flip(tgt_ids, (1,))

    # 4) concat & score
    #input_ids = t.cat((ctx_ids, tgt_ids), dim=1).to(model.device)
    device = next(model.parameters()).device
    input_ids = t.cat((ctx_ids, tgt_ids), dim=1).to(device)
    return compute_token_probabilities(input_ids, model, tokenizer, ctx_len, tgt_len)


def compute_token_probabilities(input_ids, model, tokenizer, context_len, target_len):
    """
    Compute detailed token-level probabilities, sequence log-probability,
    normalized log-probability, and perplexity for a given concatenated input.

    Args:
        input_ids (torch.Tensor): Tensor of shape [1, L] containing token IDs for context+target.
        model: A causal language model (e.g., GPT) returning logits over the vocabulary.
        tokenizer: Corresponding tokenizer to decode tokens.
        context_len (int): Number of tokens in the context portion of input_ids.
        target_len (int): Number of tokens in the target portion of input_ids.

    Returns:
        dict with keys:
            'token_log_probs': List of dicts {token, token_id, log_prob} for each target token.
            'sequence_log_prob': Sum of log-probabilities for all target tokens.
            'normalized_log_prob': Average log-prob per target token.
            'perplexity': Exponential of negative average log-prob (i.e., model perplexity).
    """
    # 1) Forward pass under no_grad + mixed precision for efficiency
    with t.no_grad(), autocast():
        outputs = model(input_ids)
        logits = outputs.logits  # Shape: [1, L, V] where V = vocab size

    token_probs = []  # Will accumulate per-token log-prob info
    # 2) Iterate over each target token position
    for i in range(context_len - 1, context_len + target_len - 1):
        # logits at position i predicts token at i+1
        next_logits = logits[0, i, :]            # [V]-shaped vector of next-token logits
        next_id = input_ids[0, i + 1].item()     # True next-token ID
        # Convert logits to probabilities via softmax
        probs = F.softmax(next_logits, dim=0)     # [V]
        p = probs[next_id].item()                # Probability assigned to the true token
        log_p = np.log(p)                        # Natural log-prob
        token_text = tokenizer.decode([next_id]) # Convert ID back to text for inspection
        # Store detailed info for this token
        token_probs.append({
            'token': token_text,
            'token_id': next_id,
            'log_prob': log_p
        })

    # 3) Compute sequence-level metrics
    seq_log_prob = sum(tp['log_prob'] for tp in token_probs)
    # Normalized log-prob = average log-prob across target tokens
    normalized_log_prob = seq_log_prob / len(token_probs) if token_probs else float('-inf')
    # Perplexity = exp(- average log-prob)
    perplexity = np.exp(-normalized_log_prob) if token_probs else float('inf')

    return {
        'token_log_probs': token_probs,
        'sequence_log_prob': seq_log_prob,
        'normalized_log_prob': normalized_log_prob,
        'perplexity': perplexity
    }


def calculate_scores_batch(contexts, targets, model, tokenizer, backward=False):
    """
    Batch‑wise scoring: assumes tokenizer.pad_token was set at load time.
    Returns (norm_lps, ppls) exactly as before.
    """

    # 1) Tokenize all context/target pairs at once
    batch = tokenizer(
        contexts,
        text_pair=targets,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    ).to(next(model.parameters()).device)

    input_ids      = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # 2) Figure out where context ends / target begins
    #    We can re‑tokenize each side to get lengths, or—which is cleaner—use `tokenizer` again:
    #    (If your tokenizer returns token_type_ids, you can use those instead.)
    ctx_enc = tokenizer(contexts, padding=True, truncation=True,
                        max_length=4096, return_tensors="pt")
    tgt_enc = tokenizer(targets, padding=True, truncation=True,
                        max_length=4096, return_tensors="pt")
    ctx_lens = (ctx_enc["input_ids"] != tokenizer.pad_token_id).sum(dim=1).tolist()
    tgt_lens = (tgt_enc["input_ids"] != tokenizer.pad_token_id).sum(dim=1).tolist()

    # 3) Forward once in FP16 for memory/speed
    with t.no_grad(), autocast():
        logits = model(input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits, dim=-1)

    # 4) Extract each sequence’s avg log‑prob and perplexity
    norm_lps, ppls = [], []
    for i, (cl, tl) in enumerate(zip(ctx_lens, tgt_lens)):
        if tl == 0:
            norm_lps.append(float("-inf"))
            ppls.append(float("inf"))
            continue

        # sum log‑probs of the tl target tokens
        seq_lp = 0.0
        for pos in range(cl - 1, cl + tl - 1):
            nxt = input_ids[i, pos + 1].item()
            seq_lp += log_probs[i, pos, nxt].item()

        avg_lp = seq_lp / tl
        norm_lps.append(avg_lp)
        ppls.append(float(np.exp(-avg_lp)))

    return norm_lps, ppls

# def calculate_scores_batch(contexts, targets, model, tokenizer, backward=False):
#     """
#     Compute normalized log-probabilities and perplexities for a batch of context-target pairs.

#     Args:
#         contexts (List[str]): List of context strings.
#         targets (List[str]): List of target strings to score against each context.
#         model: Causal language model (e.g., GPT) with .logits output.
#         tokenizer: Corresponding tokenizer for encoding text.
#         backward (bool): If True, reverse token order for backward model.

#     Returns:
#         norm_lps (List[float]): Average log-prob for each pair.
#         ppls (List[float]): Perplexity for each pair.
#     """
#     # Determine device from model parameters (handles DataParallel/Accelerate)
#     device = next(model.parameters()).device

#     # 1) Tokenize each context-target pair and concatenate without EOS
#     all_input_ids, all_attn_mask = [], []
#     for ctx, tgt in zip(contexts, targets):
#         # Encode context and target separately (no special tokens)
#         ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
#         tgt_ids = tokenizer.encode(tgt, add_special_tokens=False)
#         # If backward model, reverse token order
#         if backward:
#             ctx_ids, tgt_ids = ctx_ids[::-1], tgt_ids[::-1]
#         # Concatenate context and target tokens
#         ids = ctx_ids + tgt_ids
#         all_input_ids.append(ids)

#     # 2) Pad all sequences to the same length and build attention masks
#     max_len = max(len(ids) for ids in all_input_ids)
#     for ids in all_input_ids:
#         pad_len = max_len - len(ids)
#         # 1s for real tokens, 0s for padding
#         all_attn_mask.append([1] * len(ids) + [0] * pad_len)
#         # Extend the token list with pad_token_id
#         ids.extend([tokenizer.pad_token_id] * pad_len)

#     # Convert lists into tensors on the correct device
#     input_ids = t.tensor(all_input_ids, device=device)        # shape [B, L]
#     attn_mask = t.tensor(all_attn_mask, device=device)        # shape [B, L]

#     # 3) Forward pass: compute logits once for the entire batch
#     with t.no_grad(), autocast():
#         outputs = model(input_ids, attention_mask=attn_mask)
#         logits = outputs.logits  # shape [B, L, V]

#     # 4) Compute log-probabilities and perplexities per sample
#     norm_lps, ppls = [], []
#     B = input_ids.size(0)
#     for i in range(B):
#         # Recompute context and target lengths for this sample
#         ctx_len = len(tokenizer.encode(contexts[i], add_special_tokens=False))
#         tgt_len = len(tokenizer.encode(targets[i],    add_special_tokens=False))
#         # If no target tokens, return -inf / inf
#         if tgt_len == 0:
#             norm_lps.append(float("-inf"))
#             ppls.append(float("inf"))
#             continue

#         seq_lp = 0.0
#         # For each target position, compute log-prob via logsumexp
#         # logits[i, pos, :] predicts token at pos+1
#         for pos in range(ctx_len - 1, ctx_len + tgt_len - 1):
#             next_logits = logits[i, pos, :]             # shape [V]
#             next_id = input_ids[i, pos + 1].item()      # true token ID
#             # normalization constant: log-sum-exp over vocab
#             lse = t.logsumexp(next_logits, dim=-1)
#             # log-prob of the true token = logit - lse
#             logp = (next_logits[next_id] - lse).item()
#             seq_lp += logp

#         # Average log-prob across target tokens
#         avg_lp = seq_lp / tgt_len
#         norm_lps.append(avg_lp)
#         # Perplexity = exp(- avg log-prob)
#         ppls.append(math.exp(-avg_lp))

#     return norm_lps, ppls