import numpy as np
from tqdm.auto import tqdm
from nltk.tokenize import sent_tokenize

from .utils_batch_v2 import calculate_scores_batch


def linear_attribution_search(
    dataset, fo_model, fo_tokenizer, ba_model, ba_tokenizer,
    sentence_batch_size: int = None
):
    """
    Perform linear attribution search for citations.
    For each example in the dataset, split into sentences and highlights.
    Then score each candidate sentence under three scoring schemes:
      - Base (forward-normal)
      - Fo (forward-reverse)
      - Ba (backward-reverse)
    Returns a list of dicts with the best citation and scores for each scheme.
    """
    results = []

    # Iterate through dataset rows efficiently with tqdm
    for _, example in tqdm(dataset.iterrows(), total=len(dataset)):
        # Tokenize article and highlights into sentences
        article_sents = sent_tokenize(example['article'])
        highlight_sents = sent_tokenize(example['highlights'])
        if not highlight_sents:
            continue
        hl = highlight_sents[0]

        # Filter out sentences that are too short
        valid_sents = [sent for sent in article_sents if len(sent.split()) >= 3]
        if not valid_sents:
            continue

        # Prepare contexts and targets for each method
        # Base: highlight as context, candidate sentence as target
        ctx_base = [hl + " is a summary of" for _ in valid_sents]
        tgt_base = valid_sents

        # Fo: candidate sentence as context, highlight as target
        ctx_fo = [sent + " is a summary of" for sent in valid_sents]
        tgt_fo = [hl for _ in valid_sents]

        # Ba: backward model, reverse direction
        ctx_ba = ["is summarized by " + sent for sent in valid_sents]
        tgt_ba = [hl for _ in valid_sents]

        # Initialize score lists
        base_scores, base_ppls = [], []
        fo_scores, fo_ppls     = [], []
        ba_scores, ba_ppls     = [], []

        # Determine batch step size
        step = sentence_batch_size or len(valid_sents)

        # Process in batches to utilize GPU efficiently
        for i in range(0, len(valid_sents), step):
            # Base scoring
            b_scores, b_ppls = calculate_scores_batch(
                ctx_base[i:i+step], tgt_base[i:i+step],
                fo_model, fo_tokenizer, backward=False
            )
            base_scores.extend(b_scores)
            base_ppls.extend(b_ppls)

            # Fo scoring
            f_scores, f_ppls = calculate_scores_batch(
                ctx_fo[i:i+step], tgt_fo[i:i+step],
                fo_model, fo_tokenizer, backward=False
            )
            fo_scores.extend(f_scores)
            fo_ppls.extend(f_ppls)

            # Ba scoring
            r_scores, r_ppls = calculate_scores_batch(
                ctx_ba[i:i+step], tgt_ba[i:i+step],
                ba_model, ba_tokenizer, backward=True
            )
            ba_scores.extend(r_scores)
            ba_ppls.extend(r_ppls)

        # Skip if any score list is empty
        if not (base_scores and fo_scores and ba_scores):
            continue

        # Find the index of the best-scoring sentence
        best_b = max(range(len(base_scores)), key=lambda i: base_scores[i])
        best_f = max(range(len(fo_scores)),   key=lambda i: fo_scores[i])
        best_r = max(range(len(ba_scores)),   key=lambda i: ba_scores[i])

        # Append results with best citation and its metrics
        results.append({
            'id': example['id'],
            'highlight': hl,
            'base_citation': valid_sents[best_b],
            'base_score': base_scores[best_b],
            'base_perplexity': base_ppls[best_b],
            'fo_citation': valid_sents[best_f],
            'fo_score': fo_scores[best_f],
            'fo_perplexity': fo_ppls[best_f],
            'ba_citation': valid_sents[best_r],
            'ba_score': ba_scores[best_r],
            'ba_perplexity': ba_ppls[best_r]
        })

    return results


def binary_search_citation(
    highlight, article,
    model, tokenizer,
    backward=False, query_direction="normal",
    max_iterations: int = 30,
    sentence_batch_size: int = None
):
    """
    Perform binary search to find the best citation span.
    Recursively split the sentence list and compare left vs right spans.
    Stops when single sentence or max_iterations reached.
    Returns dict with citation text, score, and perplexity.
    """
    sentences = sent_tokenize(article)
    if not sentences or not highlight:
        return {'citation': '', 'score': float('-inf'), 'perplexity': float('inf')}

    def make_ctx_tgt(text_span):
        # Build context, target pair based on direction flags
        if query_direction == "normal" and not backward:
            return highlight + " is a summary of", text_span
        elif query_direction == "reverse" and not backward:
            return text_span + " is a summary of", highlight
        else:
            return "is summarized by " + text_span, highlight

    def binary_search_recursive(s, t, iteration=0):
        # Base cases
        if t < s or iteration >= max_iterations:
            return '', float('-inf'), float('inf')
        if s == t:
            # Single sentence case
            ctx, tgt = make_ctx_tgt(sentences[s])
            scores, ppls = calculate_scores_batch([ctx], [tgt], model, tokenizer, backward=backward)
            if scores:
                return sentences[s], scores[0], ppls[0]
            return '', float('-inf'), float('inf')

        # Split range in half
        mid = s + (t - s) // 2
        left_span  = ' '.join(sentences[s:mid+1])
        right_span = ' '.join(sentences[mid+1:t+1])
        left_ctx, left_tgt   = make_ctx_tgt(left_span)
        right_ctx, right_tgt = make_ctx_tgt(right_span)

        # Score both halves in one batch
        scores, _ = calculate_scores_batch(
            [left_ctx, right_ctx], [left_tgt, right_tgt], model, tokenizer, backward=backward
        )
        if len(scores) < 2:
            return '', float('-inf'), float('inf')

        # Recurse on the better scoring half
        if scores[0] >= scores[1]:
            return binary_search_recursive(s, mid, iteration + 1)
        else:
            return binary_search_recursive(mid + 1, t, iteration + 1)

    citation, score, perplexity = binary_search_recursive(0, len(sentences) - 1)
    return {
        'citation': citation,
        'score': score,
        'perplexity': perplexity
    }


def binary_search_attribution_search(
    dataset, fo_model, fo_tokenizer, ba_model, ba_tokenizer,
    max_iterations: int = 30,
    sentence_batch_size: int = None
):
    """
    Apply binary_search_citation to every example in the dataset.
    Returns a list of dicts with citations and scores for each model setting.
    """
    results = []
    for _, example in tqdm(dataset.iterrows(), total=len(dataset)):
        highlight_sents = sent_tokenize(example['highlights'])
        if not highlight_sents:
            continue
        highlight = highlight_sents[0]
        article   = example['article']

        # Run three configurations
        baseline = binary_search_citation(
            highlight, article, fo_model, fo_tokenizer,
            backward=False, query_direction="normal",
            max_iterations=max_iterations,
            sentence_batch_size=sentence_batch_size
        )
        fo = binary_search_citation(
            highlight, article, fo_model, fo_tokenizer,
            backward=False, query_direction="reverse",
            max_iterations=max_iterations,
            sentence_batch_size=sentence_batch_size
        )
        ba = binary_search_citation(
            highlight, article, ba_model, ba_tokenizer,
            backward=True,  query_direction="reverse",
            max_iterations=max_iterations,
            sentence_batch_size=sentence_batch_size
        )

        # Collect results
        results.append({
            'id': example['id'],
            'highlight': highlight,
            'base_citation': baseline['citation'],
            'base_score': baseline['score'],
            'base_perplexity': np.exp(-baseline['score']) if baseline['score'] != float('-inf') else float('inf'),
            'fo_citation': fo['citation'],
            'fo_score': fo['score'],
            'fo_perplexity': np.exp(-fo['score']) if fo['score'] != float('-inf') else float('inf'),
            'ba_citation': ba['citation'],
            'ba_score': ba['score'],
            'ba_perplexity': np.exp(-ba['score']) if ba['score'] != float('-inf') else float('inf')
        })
    return results


def exclusion_search_citation(
    highlight, article,
    model, tokenizer,
    backward=False, query_direction="normal",
    sentence_batch_size: int = None
):
    """
    Find the most critical sentence by exclusion search:
    For each sentence, exclude it from the article,
    score the remaining text, and pick the sentence whose
    exclusion lowers the score the most.
    Returns a dict with the worst sentence, its score, and perplexity.
    """
    sentences = sent_tokenize(article)
    if not sentences or not highlight:
        return {'citation': '', 'score': float('-inf'), 'perplexity': float('inf')}

    # Context/Target builder
    def make_ctx_tgt(text):
        if query_direction == "normal" and not backward:
            return highlight + " is a summary of", text
        elif query_direction == "reverse" and not backward:
            return text + " is a summary of", highlight
        else:
            return "is summarized by " + text, highlight

    all_scores, all_ppls = [], []
    step = sentence_batch_size or len(sentences)

    # Process in batches of exclusions
    for i in range(0, len(sentences), step):
        batch_ctxs, batch_tgts, idxs = [], [], []
        for j in range(i, min(i+step, len(sentences))):
            # Exclude sentence j
            excl = ' '.join(sentences[:j] + sentences[j+1:])
            ctx, tgt = make_ctx_tgt(excl)
            batch_ctxs.append(ctx)
            batch_tgts.append(tgt)
            idxs.append(j)
        # Score batch
        scores, ppls = calculate_scores_batch(batch_ctxs, batch_tgts, model, tokenizer, backward=backward)
        for j, s, p in zip(idxs, scores, ppls):
            all_scores.append((s, j))
            all_ppls.append(p)

    if not all_scores:
        return {'citation': '', 'score': float('-inf'), 'perplexity': float('inf')}

    # Select the sentence whose exclusion dropped the score most (min score)
    worst_score, worst_idx = min(all_scores, key=lambda x: x[0])
    worst_sent = sentences[worst_idx]

    # Re-score that sentence alone
    ctx, tgt = make_ctx_tgt(worst_sent)
    scs, pps = calculate_scores_batch([ctx], [tgt], model, tokenizer, backward=backward)
    return {
        'citation': worst_sent,
        'score': scs[0] if scs else float('-inf'),
        'perplexity': pps[0] if pps else float('inf')
    }


def exclusion_search_attribution_search(
    dataset, fo_model, fo_tokenizer, ba_model, ba_tokenizer,
    sentence_batch_size: int = None
):
    """
    Apply exclusion_search_citation to each example in the dataset.
    Returns a list of dicts with final citations and metrics for each config.
    """
    results = []
    for _, example in tqdm(dataset.iterrows(), total=len(dataset)):
        highlights = sent_tokenize(example['highlights'])
        if not highlights:
            continue
        highlight = highlights[0]
        article   = example['article']

        base = exclusion_search_citation(
            highlight, article, fo_model, fo_tokenizer,
            backward=False, query_direction="normal",
            sentence_batch_size=sentence_batch_size
        )
        fo  = exclusion_search_citation(
            highlight, article, fo_model, fo_tokenizer,
            backward=False, query_direction="reverse",
            sentence_batch_size=sentence_batch_size
        )
        ba  = exclusion_search_citation(
            highlight, article, ba_model, ba_tokenizer,
            backward=True,  query_direction="reverse",
            sentence_batch_size=sentence_batch_size
        )
        results.append({
            'id': example['id'],
            'highlight': highlight,
            'base_citation': base['citation'],
            'base_score': base['score'],
            'base_perplexity': np.exp(-base['score']) if base['score'] != float('-inf') else float('inf'),
            'fo_citation': fo['citation'],
            'fo_score': fo['score'],
            'fo_perplexity': np.exp(-fo['score']) if fo['score'] != float('-inf') else float('inf'),
            'ba_citation': ba['citation'],
            'ba_score': ba['score'],
            'ba_perplexity': np.exp(-ba['score']) if ba['score'] != float('-inf') else float('inf')
        })
    return results
