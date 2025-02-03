import torch
import heapq
from itertools import count
import bitstring
from tqdm import tqdm

def prefix_codes(probs: torch.Tensor) -> list[bitstring.BitArray]:
    """
    Given a probability distribution over a vocabulary, returns the prefix codes for each token using Huffman encoding, as BitArrays.
    
    Args:
        probs: A 1D tensor (or array) of probabilities for each token.
        
    Returns:
        A list of prefix codes (as bitstring.BitArray objects), where the i-th code corresponds to token i.
    """
    # Convert to a Python list (if not already) so we can iterate over it.
    if hasattr(probs, "tolist"):
        probs = probs.tolist()
    else:
        probs = list(probs)
        
    n = len(probs)
    # Edge case: If there is only one token, assign its code as BitArray("0")
    if n == 1:
        return [bitstring.BitArray(bin="0")]
    
    # Build the Huffman tree using a min-heap.
    # Each node in the tree is represented as a tuple: (token, left, right)
    # For leaf nodes, token is set to an integer; for internal nodes, token is None.
    counter = count()  # Unique counter to avoid comparison issues.
    heap = []  # Heap elements: (weight, unique_id, node)
    for i, p in enumerate(probs):
        heapq.heappush(heap, (p, next(counter), (i, None, None)))
    
    while len(heap) > 1:
        w1, _ , node1 = heapq.heappop(heap)
        w2, _ , node2 = heapq.heappop(heap)
        new_weight = w1 + w2
        # Create an internal node; token is None.
        new_node = (None, node1, node2)
        heapq.heappush(heap, (new_weight, next(counter), new_node))
    
    # The remaining element in heap is the root of the Huffman tree.
    root = heap[0][2]
    codes = [None] * n

    # Traverse the Huffman tree to assign binary codes:
    def traverse(node, code):
        token, left, right = node
        if token is not None:
            # Leaf node: assign the current code (or "0" if code is empty)
            literal = code if code != "" else "0"
            codes[token] = bitstring.BitArray(bin=literal)
            return
        if left is not None:
            traverse(left, code + "0")
        if right is not None:
            traverse(right, code + "1")
    
    traverse(root, "")
    return codes


def encode_tokens(token_ids: list[int], probs_list: list[torch.Tensor]) -> bitstring.BitArray:
    """
    Encodes a sequence of token ids using Huffman encoding for each corresponding probability distribution.
    
    For each probability distribution in probs_list, a prefix code (codebook) is computed using Huffman encoding.
    Then, given the token id, its corresponding Huffman code is appended to the final result.
    
    Args:
        token_ids: A list of token ids corresponding to each prediction.
        probs_list: A list of probability distributions (each as a 1D torch.Tensor) corresponding to each token.
        
    Returns:
        A bitstring.BitArray object containing the concatenated Huffman encoded sequence.
    """
    result = bitstring.BitArray()
    for token_id, probs in tqdm(zip(token_ids, probs_list), total=len(token_ids), desc="Encoding tokens"):
        # Compute the Huffman codebook for the given probability distribution.
        codebook = prefix_codes(probs)
        if token_id < 0 or token_id >= len(codebook):
            raise ValueError("Token id is out of bounds for the provided probability distribution.")
        # Append the Huffman code for the specified token.
        result.append(codebook[token_id])
    return result
