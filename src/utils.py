import torch as t
import numpy as np
import torch.nn.functional as F

def calculate_score(query, answer, model, tokenizer, backward=False, query_direction="reverse", task='citation', debug=False):
    """
    Calculate log probability of response given prompt or vice versa based on query_direction.

    Args:
        query_info (dict): Dictionary containing query, answer, and task type
        model: The language model
        tokenizer: The corresponding tokenizer
        query_direction (str): "reverse" for response->query, "normal" for query->response
        debug (bool): If True, print debug information

    Returns:
        dict: Contains token-wise and sequence log probabilities
    """
    
    # First, prepare the texts
    if query_direction == "normal":
        assert backward == False, "normal query direction does not support backward model - only for forward model"
        
        conditioning_prompt = 'is a summary of' if task =='citation' else 'has an answer to'
        # there's no discussion on the conditioning prompt for this, so we assume its:
        # context [query + conditioning prompt] and target [answer]
        # where conditioning prompt -> "is summarized by"
        # e.g. "Harry potter star... gains access to reported... __ is summarized by __" -> ""
        
        context = query + conditioning_prompt
        target = answer
        
    elif query_direction == "reverse":
        if not backward: # Forward model (Fo) - Algorithm 3
            conditioning_prompt = 'is a summary of' if task =='citation' else 'has an answer to'
            context = answer + conditioning_prompt
            target = query
        else: # Backward model (Ba) - Algorithm 2
            conditioning_prompt = 'is summarized by' if task =='citation' else 'is answered by'
            context = conditioning_prompt + answer
            target = query
            
    # DEBUG
    if debug:
        print(f"Settings: {backward} {query_direction}")
        print(f"Context: {context}")
        print(f"Target: {target}")
        print(f"Full sentence: {context + target if not backward else target + context}")
        print("\n")

    
    # tokenize
    context_ids = tokenizer.encode(context, return_tensors="pt")
    target_ids = tokenizer.encode(target, return_tensors="pt")
    
    # Store length to "divide" the texts later
    context_len = context_ids.shape[1]
    target_len = target_ids.shape[1]
    
    # reverse the token for backward
    if backward:
        context_ids = t.flip(context_ids, (1,))
        target_ids = t.flip(target_ids, (1,))
        
        
    # Combine the context and target tokens
    input_ids = t.cat((context_ids, target_ids), dim=1).to(model.device)
    
    # regardless of the query direction and backward/forward model, the scoring is the same:
    return compute_token_probabilities(input_ids, model, tokenizer, context_len, target_len)



def compute_token_probabilities(input_ids, model, tokenizer, context_len, target_len):
    """
    Compute token-wise and sequence probabilities from model outputs.

    Args:
        input_ids: Combined context and target token IDs
        model: The language model
        tokenizer: The corresponding tokenizer
        context_len: Length of the context tokens
        target_len: Length of the target tokens

    Returns:
        dict: Contains token-wise and sequence log probabilities
    """
    # Get model output
    with t.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Extract token probabilities for the target text
    token_probs = []
    # Process tokens in the target portion of the combined input
    for i in range(context_len - 1, context_len + target_len - 1):
        # Get the logits for the current position
        next_token_logits = logits[0, i, :]  # no batch, sequence i, all vocab

        # Get the actual token that should follow
        next_token_id = input_ids[0, i+1].item()

        # Convert logits to probabilities
        next_token_probs = F.softmax(next_token_logits, dim=0)
        prob = next_token_probs[next_token_id].item()
        log_prob = np.log(prob)

        token_text = tokenizer.decode([next_token_id])
        token_probs.append({
            'token': token_text,
            'token_id': next_token_id,
            'log_prob': log_prob
        })

    # Calculate sequence probability
    sequence_log_prob = sum(tp['log_prob'] for tp in token_probs)
    # Normalize by length to get per-token average
    normalized_log_prob = sequence_log_prob / len(token_probs)
    # Convert to perplexity if needed
    perplexity = np.exp(-sequence_log_prob / len(token_probs))

    return {
        'token_log_probs': token_probs,
        'sequence_log_prob': sequence_log_prob,
        'normalized_log_prob': normalized_log_prob,
        'perplexity': perplexity
    }

## maybe add tfidf score?


# wrapper
# def calculate_llm_score(query, answer, model, tokenizer, task='citation', backward=False, debug=False):
#     """
#     Calculate log probability of response given prompt or vice versa.

#     Args:
#         query (str): The prompt text
#         answer (str): The response text
#         model: The language model
#         tokenizer: The corresponding tokenizer
#         task (str): The task type, either 'citation' or something else
#         backward (bool): If True, use backward conditioning, otherwise forward
#         debug (bool): If True, print debug information

#     Returns:
#         dict: Contains token-wise and sequence log probabilities
#     """
#     query_info = create_query_answer_prompt(query, answer, task, backward)
#     scores = calculate_score(query_info, model, tokenizer, debug)
#     return scores

