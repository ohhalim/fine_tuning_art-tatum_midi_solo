"""
Data Collator for Music Transformer

Industry-standard batching with HuggingFace data collators
"""

import torch
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class MusicDataCollator:
    """
    Data collator for batching MIDI sequences

    Handles:
    - Dynamic padding (pad to longest in batch)
    - Attention masks
    - Label preparation for causal LM

    Compatible with HuggingFace Trainer
    """

    pad_token_id: int = 0
    pad_to_multiple_of: int = 8  # For efficiency on GPUs

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Batch a list of examples

        Args:
            features: List of dicts with 'input_ids', 'attention_mask', 'labels'

        Returns:
            Batched dict with padded tensors
        """
        # Get max length in batch
        max_len = max(len(f['input_ids']) for f in features)

        # Pad to multiple of pad_to_multiple_of for efficiency
        if self.pad_to_multiple_of > 0:
            max_len = ((max_len + self.pad_to_multiple_of - 1)
                      // self.pad_to_multiple_of * self.pad_to_multiple_of)

        # Prepare batch tensors
        batch_size = len(features)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # -100 ignored in loss

        # Fill tensors
        for i, feature in enumerate(features):
            seq_len = len(feature['input_ids'])

            # Input IDs
            if isinstance(feature['input_ids'], torch.Tensor):
                input_ids[i, :seq_len] = feature['input_ids']
            else:
                input_ids[i, :seq_len] = torch.tensor(feature['input_ids'])

            # Attention mask
            if isinstance(feature['attention_mask'], torch.Tensor):
                attention_mask[i, :seq_len] = feature['attention_mask']
            else:
                attention_mask[i, :seq_len] = torch.tensor(feature['attention_mask'])

            # Labels (shifted in model forward, so just copy input_ids)
            if 'labels' in feature:
                if isinstance(feature['labels'], torch.Tensor):
                    labels[i, :seq_len] = feature['labels']
                else:
                    labels[i, :seq_len] = torch.tensor(feature['labels'])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


if __name__ == "__main__":
    print("Testing MusicDataCollator...\n")

    # Create collator
    collator = MusicDataCollator(pad_token_id=0)

    # Create sample features (different lengths)
    features = [
        {
            'input_ids': torch.tensor([1, 10, 20, 30, 2]),
            'attention_mask': torch.tensor([1, 1, 1, 1, 1]),
            'labels': torch.tensor([1, 10, 20, 30, 2])
        },
        {
            'input_ids': torch.tensor([1, 15, 25, 2]),
            'attention_mask': torch.tensor([1, 1, 1, 1]),
            'labels': torch.tensor([1, 15, 25, 2])
        },
        {
            'input_ids': torch.tensor([1, 5, 10, 15, 20, 25, 2]),
            'attention_mask': torch.tensor([1, 1, 1, 1, 1, 1, 1]),
            'labels': torch.tensor([1, 5, 10, 15, 20, 25, 2])
        }
    ]

    print(f"Input sequences:")
    for i, f in enumerate(features):
        print(f"  Sequence {i}: length={len(f['input_ids'])}")

    # Collate
    batch = collator(features)

    print(f"\nBatched tensors:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")

    print(f"\ninput_ids:")
    print(batch['input_ids'])
    print(f"\nattention_mask:")
    print(batch['attention_mask'])

    print("\n✅ Data collator test passed!")
