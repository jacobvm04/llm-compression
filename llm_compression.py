from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn import functional as F
from tqdm import tqdm, trange

from huffman import encode_tokens, prefix_codes

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

def compute_probs(input_str):
    """
    Computes the next-token probability distributions for the input string by iteratively reconstructing the context.
    For each position after the seed token, it computes:
      - the probability distribution for the next token,
      - the token id of the actual next token.
    
    Returns:
        A tuple of (tokens, token_ids, probs_list) where:
            tokens: the list of tokens from the input string.
            token_ids: a list of token ids corresponding to each predicted token (excluding the seed).
            probs_list: a list of probability distributions (each as a 1D torch.Tensor) for each prediction.
    """
    inputs = tokenizer(input_str, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    token_ids = []
    probs_list = []
    for i in trange(1, len(tokens), desc="Computing probabilities"):
        context_str = tokenizer.convert_tokens_to_string(tokens[:i])
        probs = next_token_probs(context_str)
        probs_list.append(probs)
        target_id = tokenizer.convert_tokens_to_ids(tokens[i])
        token_ids.append(target_id)
    return tokens, token_ids, probs_list

def compute_next_token(input_str, rank):
    """
    Computes the token following the input string whose probability has the given rank.
    It uses the next_token_probs helper to obtain the next-token distribution.
    """
    probs = next_token_probs(input_str)
    sorted_indices = torch.argsort(probs, descending=True)

    if rank < 1 or rank > sorted_indices.shape[0]:
        raise ValueError("Rank is out of bounds. It should be between 1 and the vocabulary size.")

    selected_token_id = sorted_indices[rank - 1].item()
    selected_token = tokenizer.convert_ids_to_tokens(selected_token_id)
    return selected_token

## Helper function
def next_token_probs(context_str):
    """
    Given a context string, tokenizes it and returns the softmax probabilities
    for the next token prediction.
    """
    inputs_context = tokenizer(context_str, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(inputs_context.input_ids, attention_mask=inputs_context.attention_mask)
    logits = outputs.logits[0, -1, :]  # shape: (vocab_size,)
    probs = F.softmax(logits, dim=-1)
    return probs

def encode(input_str):
    """
    Encodes the input string into a tuple of (seed, encoded_bitstring).
    The seed is the first token from the input.
    Uses compute_probs to obtain token probabilities and ranks, and then encodes the ranks using Huffman encoding.
    """
    if len(input_str) < 1:
        raise ValueError("Input string must be at least 1 character long.")

    tokens, token_ids, probs_list = compute_probs(input_str)
    seed = tokens[0]
    encoded = encode_tokens(token_ids, probs_list)
    return seed, encoded

def decode(seed, encoded):
    """
    Decodes the original string from a given seed token and the encoded bitstring.
    
    The decoding process iteratively:
      - Computes the probability distribution for the current context,
      - Builds the Huffman codebook for that distribution,
      - Decodes the next token by matching the beginning of the encoded bitstring against the codebook,
      - Removes the matched code from the bitstring and appends the decoded token to the context.
    This repeats until the encoded bitstring is exhausted.
    """
    decoded_tokens = [seed]
    current_str = seed
    
    total_bits = len(encoded)
    with tqdm(total=total_bits, desc="Decoding tokens") as pbar:
        while len(encoded) > 0:
            # Compute next-token probability distribution for the current context.
            probs = next_token_probs(current_str)
            # Build the Huffman codebook for the distribution.
            codes = prefix_codes(probs)
            # Identify which token's code is a prefix of the current encoded bitstring.
            matched_token_id = None
            for token_id, code in enumerate(codes):
                if encoded[:len(code)] == code:
                    matched_token_id = token_id
                    break
            if matched_token_id is None:
                raise ValueError("No matching token found in the encoded bitstring.")
            # Remove the matched code from the beginning of the encoded bitstring.
            bits_consumed = len(codes[matched_token_id])
            encoded = encoded[bits_consumed:]
            pbar.update(bits_consumed)
            # Convert the token id to the corresponding token string.
            token = tokenizer.convert_ids_to_tokens(matched_token_id)
            decoded_tokens.append(token)
            # Update the current context.
            current_str = tokenizer.convert_tokens_to_string(decoded_tokens)
    
    return current_str.replace("Ġ", " ")
