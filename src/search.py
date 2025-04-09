from collections import defaultdict

import numpy as np
from nltk.tokenize import sent_tokenize
from tqdm.auto import tqdm

from .utils import calculate_score, calculate_score_batch


def linear_attribution_search(dataset, fo_model, fo_tokenizer, ba_model, ba_tokenizer):
    """
    Perform linear attribution search for citations as described in TRLM paper.

    For each highlight (summary sentence), find the most likely article sentence
    that it was derived from by scoring all possible pairs.
    """
    results = []

    # Process only the first few examples for demonstration
    for idx, example in tqdm(dataset.iterrows(), total=len(dataset)):
        
        # Split article and highlights into sentences
        article_sentences = sent_tokenize(example['article'])
        highlight_sentences = sent_tokenize(example['highlights'])

        # For demonstration, process just the first highlight sentence
        if not highlight_sentences:
            continue

        highlight = highlight_sentences[0]

        # Store best attribution for each model
        best_ba_sentence = None
        best_ba_score = float('-inf')
        best_fo_sentence = None
        best_fo_score = float('-inf')
        best_base_sentence = None
        best_base_score = float('-inf')

        # Linear search through all article sentences
        for sentence in article_sentences:
            # Skip very short sentences
            if len(sentence.split()) < 3:
                continue

            # Calculate scores using both models
            base_score = calculate_score(
                highlight, sentence, fo_model, fo_tokenizer, backward=False, query_direction="normal")
            fo_score = calculate_score(
                highlight, sentence, fo_model, fo_tokenizer, backward=False, query_direction="reverse")
            ba_score = calculate_score(
                highlight, sentence, ba_model, ba_tokenizer, backward=True, query_direction="reverse")

            # Track best scores
            if base_score['normalized_log_prob'] > best_base_score:
                best_base_score = base_score['normalized_log_prob']
                best_base_sentence = sentence

            if ba_score['normalized_log_prob'] > best_ba_score:
                best_ba_score = ba_score['normalized_log_prob']
                best_ba_sentence = sentence

            if fo_score['normalized_log_prob'] > best_fo_score:
                best_fo_score = fo_score['normalized_log_prob']
                best_fo_sentence = sentence

        # Add results to our list
        results.append({
            'id': example['id'],
            'highlight': highlight,
            'base_citation': best_base_sentence,
            'base_score': best_base_score,
            'base_perplexity': np.exp(-best_base_score),
            'ba_citation': best_ba_sentence,
            'ba_score': best_ba_score,
            'ba_perplexity': np.exp(-best_ba_score),
            'fo_citation': best_fo_sentence,
            'fo_score': best_fo_score,
            'fo_perplexity': np.exp(-best_fo_score)
        })

    return results



def linear_attribution_search_batched(pairs, meta, fo_model, fo_tokenizer, ba_model, ba_tokenizer, batch_size=32):
    """
    Perform batched linear attribution search from a pre-tokenized (highlight, article_sentence) list.

    Args:
        pairs (list of tuples): (highlight, article_sentence)
        meta (list of dicts): metadata including 'id', 'highlight', 'article_sentence', 'article_idx'
        fo_model, ba_model: Forward and backward LMs
        fo_tokenizer, ba_tokenizer: Corresponding tokenizers
        batch_size (int): Batch size for scoring

    Returns:
        List of dicts with best citation match for each article (by model)
    """

    # Unpack query/answer pairs
    highlights = [q for q, _ in pairs]
    article_sents = [a for _, a in pairs]

    def batch_score_all(model, tokenizer, backward, direction):
        scores = []
        for i in tqdm(range(0, len(pairs), batch_size)):
            batch_q = highlights[i:i+batch_size]
            batch_a = article_sents[i:i+batch_size]
            batch_scores = calculate_score_batch(
                batch_q, batch_a, model, tokenizer,
                backward=backward, query_direction=direction
            )
            scores.extend(batch_scores)
        return scores

    # Compute all three sets of scores
    base_scores = batch_score_all(fo_model, fo_tokenizer, backward=False, direction="normal")
    fo_scores   = batch_score_all(fo_model, fo_tokenizer, backward=False, direction="reverse")
    ba_scores   = batch_score_all(ba_model, ba_tokenizer, backward=True,  direction="reverse")

    # Group and aggregate best scores per article ID
    grouped = defaultdict(lambda: {
        'highlight': None,
        'id': None,
        'base': (-float('inf'), None),
        'fo':   (-float('inf'), None),
        'ba':   (-float('inf'), None),
    })

    for i, m in enumerate(meta):
        ex_id = m['id']
        hl = m['highlight']
        sent = m['article_sentence']

        grouped[ex_id]['id'] = ex_id
        grouped[ex_id]['highlight'] = hl

        if base_scores[i]['normalized_log_prob'] > grouped[ex_id]['base'][0]:
            grouped[ex_id]['base'] = (base_scores[i]['normalized_log_prob'], sent)

        if fo_scores[i]['normalized_log_prob'] > grouped[ex_id]['fo'][0]:
            grouped[ex_id]['fo'] = (fo_scores[i]['normalized_log_prob'], sent)

        if ba_scores[i]['normalized_log_prob'] > grouped[ex_id]['ba'][0]:
            grouped[ex_id]['ba'] = (ba_scores[i]['normalized_log_prob'], sent)

    # Final results
    results = []
    for ex_id, info in grouped.items():
        results.append({
            'id': info['id'],
            'highlight': info['highlight'],
            'base_citation': info['base'][1],
            'base_score': info['base'][0],
            'base_perplexity': np.exp(-info['base'][0]),
            'fo_citation': info['fo'][1],
            'fo_score': info['fo'][0],
            'fo_perplexity': np.exp(-info['fo'][0]),
            'ba_citation': info['ba'][1],
            'ba_score': info['ba'][0],
            'ba_perplexity': np.exp(-info['ba'][0]),
        })

    return results


def binary_search_citation(highlight, article, model, tokenizer, backward=False, query_direction="normal", max_iterations=30):
    # Split the article into individual sentences using NLTK's sentence tokenizer
    sentences = sent_tokenize(article)
    if not sentences:  # Return default values if no sentences are found in the article
        return {'citation': '', 'score': float('-inf'), 'perplexity': float('inf')}
    
    # Split the highlight into sentences and use only the first sentence
    # highlight_sentences = sent_tokenize(highlight)
    # if not highlight_sentences:  # Return default values if no sentences are found in the highlight
    #     return {'citation': '', 'score': float('-inf'), 'perplexity': float('inf')}
    # highlight = highlight_sentences[0]  # Select the first sentence of the highlight
    
    # because only the first highlight sentence is passed, no need to retokenize
    if not highlight:
        return {'citation': '', 'score': float('-inf'), 'perplexity': float('inf')}

    
    # Define a recursive binary search function to find the best citation
    def binary_search_recursive(s, t, iteration=0):
        if t - s <= 0 or iteration >= max_iterations:  # Base case: if the search range is invalid or max iterations are reached
            if t < s:  # Return default values if the range is invalid
                return '', float('-inf'), float('inf')
            # Combine sentences in the range [s, t] into a single string
            a_half = ' '.join(sentences[s:t + 1])
            # Calculate the LLM score for the combined text using calculate_score with query_direction parameter
            result = calculate_score(highlight, a_half, model, tokenizer, backward=backward, query_direction=query_direction)
            score = result['normalized_log_prob']
            perplexity = result['perplexity']
            return a_half, score, perplexity
        
        # Calculate the midpoint of the current range
        mid = s + (t - s) // 2
        # Split the sentences into two halves: [s, mid] and [mid+1, t]
        a_half1 = ' '.join(sentences[s:mid + 1])
        a_half2 = ' '.join(sentences[mid + 1:t + 1])
        # Calculate LLM scores for both halves
        result1 = calculate_score(highlight, a_half1, model, tokenizer, backward=backward, query_direction=query_direction)
        result2 = calculate_score(highlight, a_half2, model, tokenizer, backward=backward, query_direction=query_direction)
        s1, s2 = result1['normalized_log_prob'], result2['normalized_log_prob']
        
        # Debugging
        # print(f"Binary Search (Backward={backward}): s={s}, t={t}, Mid={mid}, s1={s1}, s2={s2}")
        
        if s1 > s2:
            return binary_search_recursive(s, mid, iteration + 1)
        else:
            return binary_search_recursive(mid + 1, t, iteration + 1)
    
    s, t = 0, len(sentences) - 1
    citation, score, perplexity = binary_search_recursive(s, t)
    
    if not citation:
        return {'citation': '', 'score': float('-inf'), 'perplexity': float('inf')}
    
    return {
        'citation': citation,
        'score': score,
        'perplexity': perplexity
    }
    
    
def binary_search_attribution_search(dataset, fo_model, fo_tokenizer, ba_model, ba_tokenizer, max_iterations=30):
    """
    For each example in the dataset, perform binary search attribution for citations.
    
    Returns a list of dictionaries with keys:
        'id', 'highlight',
        'base_citation', 'base_score', 'base_perplexity',
        'fo_citation', 'fo_score', 'fo_perplexity',
        'ba_citation', 'ba_score', 'ba_perplexity'
    """
    results = []
    for idx, example in tqdm(dataset.iterrows(), total=len(dataset)):
        # Split article and highlights into sentences
        article = example['article']
        
        highlight_sentences = sent_tokenize(example['highlights'])
        if not highlight_sentences:
            continue
        highlight = highlight_sentences[0]
        
        # Baseline: Using forward baseline (query_direction="normal", backward=False)
        baseline_result = binary_search_citation(highlight, article, fo_model, fo_tokenizer, backward=False, query_direction="normal", max_iterations=max_iterations)
        
        # TRLM-Fo: Using forward model with query_direction="reverse" (A->S direction)
        fo_result = binary_search_citation(highlight, article, fo_model, fo_tokenizer, backward=False, query_direction="reverse", max_iterations=max_iterations)
        
        # TRLM-Ba: Using backward model with query_direction="reverse" (A->S direction)
        ba_result = binary_search_citation(highlight, article, ba_model, ba_tokenizer, backward=True, query_direction="reverse", max_iterations=max_iterations)
        
        results.append({
            'id': example['id'],
            'highlight': highlight,
            'base_citation': baseline_result['citation'],
            'base_score': baseline_result['score'],
            'base_perplexity': np.exp(-baseline_result['score']),
            'fo_citation': fo_result['citation'],
            'fo_score': fo_result['score'],
            'fo_perplexity': np.exp(-fo_result['score']),
            'ba_citation': ba_result['citation'],
            'ba_score': ba_result['score'],
            'ba_perplexity': np.exp(-ba_result['score'])
        })
    
    return results


def exclusion_search_citation(highlight, article, model, tokenizer, backward=False, query_direction="normal"):
    # 1. Split the entire sentence group into individual sentences
    sentences = sent_tokenize(article)
    if not sentences:
        return {'citation': '', 'score': float('-inf'), 'perplexity': float('inf')}
    
    # Split the highlight into sentences and use only the first sentence
    # highlight_sentences = sent_tokenize(highlight)
    # if not highlight_sentences:
    #     return {'citation': '', 'score': float('-inf'), 'perplexity': float('inf')}
    # highlight = highlight_sentences[0]  # Use only the first sentence
    
    
    # 
    
    # 2. Calculate calculate_llm_score for each sentence group with one sentence removed from index 0 to the end
    all_scores = []
    for i in range(len(sentences)):
        # Create a sentence group excluding the sentence at index i
        excluded_sentences = sentences[:i] + sentences[i+1:]
        if not excluded_sentences:
            score = float('-inf')  # Minimum score for an empty set
            perplexity = float('inf')
        else:
            # Combine the remaining sentences into one (maintain context)
            combined_text = " ".join(excluded_sentences)
            result = calculate_score(highlight, combined_text, model, tokenizer, backward=backward, query_direction=query_direction)
            score = result['normalized_log_prob']
            perplexity = result['perplexity']
        all_scores.append((score, perplexity, i))
    
    if not all_scores:
        return {'citation': '', 'score': float('-inf'), 'perplexity': float('inf')}
    
    # 3. Select the sentence that results in the lowest relevance when removed
    worst_score, worst_perplexity, worst_idx = min(all_scores, key=lambda x: x[0])
    worst_citation = sentences[worst_idx]  # Sentence with the lowest relevance when removed
    
    # 4. Calculate the individual score for the selected sentence
    individual_result = calculate_score(worst_citation, highlight, model, tokenizer, backward=backward, query_direction=query_direction)
    individual_score = individual_result['normalized_log_prob']
    individual_perplexity = individual_result['perplexity']
    
    #print(f"All scores (excluding each sentence): {[score for score, _, _ in all_scores]}")
    #print(f"Worst score: {worst_score}, Perplexity: {worst_perplexity}, Sentence index: {worst_idx}")
    #print(f"Individual score for selected citation: {individual_score}, Individual perplexity: {individual_perplexity}")
    
    return {
        'citation': worst_citation,
        'score': individual_score,
        'perplexity': individual_perplexity
    }


def exclusion_search_attribution_search(dataset, fo_model, fo_tokenizer, ba_model, ba_tokenizer):
    
    results = []
    for idx, example in tqdm(dataset.iterrows(), total=len(dataset)):
        article = example['article']
        
        # get only the first sentence
        highlight_sentences = sent_tokenize(example['highlights'])
        if not highlight_sentences:
            continue
        highlight = highlight_sentences[0]
        
        # Baseline: forward 모델, backward=False, query_direction="normal"
        baseline_result = exclusion_search_citation(highlight, article, fo_model, fo_tokenizer, backward=False, query_direction="normal")
        # Forward (fo): forward 모델, backward=False, query_direction="reverse"
        fo_result = exclusion_search_citation(highlight, article, fo_model, fo_tokenizer, backward=False, query_direction="reverse")
        # Backward (ba): backward 모델, backward=True, query_direction="reverse"
        ba_result = exclusion_search_citation(highlight, article, ba_model, ba_tokenizer, backward=True, query_direction="reverse")
        
        results.append({
            'id': example['id'],
            'highlight': highlight,
            'base_citation': baseline_result['citation'],
            'base_score': baseline_result['score'],
            'base_perplexity': np.exp(-baseline_result['score']),
            'fo_citation': fo_result['citation'],
            'fo_score': fo_result['score'],
            'fo_perplexity': np.exp(-fo_result['score']),
            'ba_citation': ba_result['citation'],
            'ba_score': ba_result['score'],
            'ba_perplexity': np.exp(-ba_result['score'])
        })
    
    return results