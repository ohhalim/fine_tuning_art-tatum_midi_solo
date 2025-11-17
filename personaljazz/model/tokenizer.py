"""
Simple text tokenizer for PersonalJazz

In production, use a proper tokenizer like SentencePiece or BPE
"""

import torch


# Simple vocabulary (demo purposes)
# In production, load from pre-trained tokenizer
VOCAB = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}

# Add common jazz/music terms
COMMON_WORDS = [
    'jazz', 'piano', 'style', 'improvisation', 'modal', 'bebop', 'swing',
    'blues', 'chord', 'progression', 'melody', 'rhythm', 'tempo', 'slow',
    'fast', 'medium', 'bill', 'evans', 'bud', 'powell', 'art', 'tatum',
    'ohhalim', 'personal', 'solo', 'ballad', 'uptempo', 'waltz'
]

for i, word in enumerate(COMMON_WORDS):
    VOCAB[word] = i + 4


def tokenize_text(text: str, max_length: int = 128, device: str = 'cuda'):
    """
    Simple word-level tokenization

    Args:
        text: Input text
        max_length: Maximum sequence length
        device: Device to place tensors

    Returns:
        token_ids: Token IDs (1, T)
        attention_mask: Attention mask (1, T)
    """
    # Lowercase and split
    words = text.lower().split()

    # Convert to IDs
    token_ids = [VOCAB.get('<START>')]
    for word in words[:max_length - 2]:  # Reserve space for START and END
        token_ids.append(VOCAB.get(word, VOCAB.get('<UNK>')))
    token_ids.append(VOCAB.get('<END>'))

    # Pad to max_length
    attention_mask = [1] * len(token_ids)
    while len(token_ids) < max_length:
        token_ids.append(VOCAB.get('<PAD>'))
        attention_mask.append(0)

    # Convert to tensors
    token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=device)

    return token_ids, attention_mask


def detokenize(token_ids: torch.Tensor):
    """Convert token IDs back to text"""
    inv_vocab = {v: k for k, v in VOCAB.items()}
    words = []

    for token_id in token_ids:
        token_id = token_id.item()
        word = inv_vocab.get(token_id, '<UNK>')
        if word in ['<PAD>', '<START>', '<END>']:
            continue
        words.append(word)

    return ' '.join(words)
