import numpy as np
from tqdm.auto import tqdm

from nltk.tokenize import sent_tokenize
from .utils import calculate_score


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
