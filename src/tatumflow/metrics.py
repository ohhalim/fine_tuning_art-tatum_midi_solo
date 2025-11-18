"""
Evaluation metrics for music generation
Implements objective metrics from ImprovNet and AMT papers
"""

import torch
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional
import warnings


class MusicMetrics:
    """Collection of objective metrics for music evaluation"""

    @staticmethod
    def pitch_class_distribution(tokens: torch.Tensor, tokenizer) -> np.ndarray:
        """
        Compute 12-dimensional pitch class histogram

        Args:
            tokens: Token tensor (can be 1D or 2D)
            tokenizer: TatumFlowTokenizer instance

        Returns:
            12-d array with pitch class distribution
        """
        if tokens.dim() == 2:
            tokens = tokens[0]  # Take first batch

        pc_dist = np.zeros(12)

        for token_id in tokens:
            token_str = tokenizer.id_to_token.get(token_id.item(), "")
            if token_str.startswith("NOTE_ON_"):
                pitch = int(token_str.split("_")[2])
                pc = pitch % 12
                pc_dist[pc] += 1

        # Normalize
        pc_dist = pc_dist / (pc_dist.sum() + 1e-6)
        return pc_dist

    @staticmethod
    def pitch_class_kl_divergence(
        tokens_gen: torch.Tensor,
        tokens_ref: torch.Tensor,
        tokenizer
    ) -> float:
        """
        KL divergence between pitch class distributions
        Lower is better (more similar to reference)

        Args:
            tokens_gen: Generated tokens
            tokens_ref: Reference tokens
            tokenizer: Tokenizer instance

        Returns:
            KL divergence value
        """
        pc_gen = MusicMetrics.pitch_class_distribution(tokens_gen, tokenizer)
        pc_ref = MusicMetrics.pitch_class_distribution(tokens_ref, tokenizer)

        # Add small constant to avoid log(0)
        pc_gen = pc_gen + 1e-10
        pc_ref = pc_ref + 1e-10

        # KL(P||Q) = sum(P * log(P/Q))
        kl = entropy(pc_ref, pc_gen)

        return float(kl)

    @staticmethod
    def pitch_class_transition_matrix(tokens: torch.Tensor, tokenizer) -> np.ndarray:
        """
        Compute 12x12 matrix of pitch class transitions

        Args:
            tokens: Token tensor
            tokenizer: Tokenizer instance

        Returns:
            12x12 transition matrix
        """
        if tokens.dim() == 2:
            tokens = tokens[0]

        pctm = np.zeros((12, 12))
        last_pc = None

        for token_id in tokens:
            token_str = tokenizer.id_to_token.get(token_id.item(), "")
            if token_str.startswith("NOTE_ON_"):
                pitch = int(token_str.split("_")[2])
                pc = pitch % 12

                if last_pc is not None:
                    pctm[last_pc, pc] += 1

                last_pc = pc

        # Normalize rows
        row_sums = pctm.sum(axis=1, keepdims=True) + 1e-6
        pctm = pctm / row_sums

        return pctm

    @staticmethod
    def pctm_cosine_similarity(
        tokens_gen: torch.Tensor,
        tokens_ref: torch.Tensor,
        tokenizer
    ) -> float:
        """
        Cosine similarity between pitch class transition matrices
        Higher is better (1.0 = identical transitions)

        Args:
            tokens_gen: Generated tokens
            tokens_ref: Reference tokens
            tokenizer: Tokenizer instance

        Returns:
            Cosine similarity (0-1)
        """
        pctm_gen = MusicMetrics.pitch_class_transition_matrix(tokens_gen, tokenizer)
        pctm_ref = MusicMetrics.pitch_class_transition_matrix(tokens_ref, tokenizer)

        # Flatten to 144-d vectors
        pctm_gen_flat = pctm_gen.flatten()
        pctm_ref_flat = pctm_ref.flatten()

        # Cosine similarity
        similarity = 1 - cosine(pctm_gen_flat, pctm_ref_flat)

        return float(similarity)

    @staticmethod
    def note_density(
        tokens: torch.Tensor,
        tokenizer,
        segment_length_sec: float = 5.0
    ) -> float:
        """
        Average number of notes per time segment

        Args:
            tokens: Token tensor
            tokenizer: Tokenizer instance
            segment_length_sec: Length of segment in seconds

        Returns:
            Notes per segment
        """
        if tokens.dim() == 2:
            tokens = tokens[0]

        note_count = 0
        total_time_ms = 0

        for token_id in tokens:
            token_str = tokenizer.id_to_token.get(token_id.item(), "")

            if token_str.startswith("NOTE_ON_"):
                note_count += 1
            elif token_str.startswith("TIME_"):
                time_steps = int(token_str.split("_")[1])
                total_time_ms += time_steps * tokenizer.config.time_quantization_ms

        total_time_sec = total_time_ms / 1000.0
        num_segments = max(1, total_time_sec / segment_length_sec)

        return note_count / num_segments

    @staticmethod
    def average_ioi(tokens: torch.Tensor, tokenizer) -> float:
        """
        Average inter-onset interval in seconds

        Args:
            tokens: Token tensor
            tokenizer: Tokenizer instance

        Returns:
            Average IOI in seconds
        """
        if tokens.dim() == 2:
            tokens = tokens[0]

        onsets = []
        current_time_ms = 0
        chunk_start_ms = 0

        for token_id in tokens:
            token_str = tokenizer.id_to_token.get(token_id.item(), "")

            if token_str == tokenizer.config.time_shift_token:
                # New chunk, reset time
                chunk_start_ms = current_time_ms

            elif token_str.startswith("TIME_"):
                time_steps = int(token_str.split("_")[1])
                current_time_ms = chunk_start_ms + time_steps * tokenizer.config.time_quantization_ms

            elif token_str.startswith("NOTE_ON_"):
                onsets.append(current_time_ms)

        if len(onsets) < 2:
            return 0.0

        iois = np.diff(onsets) / 1000.0  # Convert to seconds
        return float(np.mean(iois))

    @staticmethod
    def unique_pitches(tokens: torch.Tensor, tokenizer) -> int:
        """
        Number of unique MIDI pitches used

        Args:
            tokens: Token tensor
            tokenizer: Tokenizer instance

        Returns:
            Number of unique pitches
        """
        if tokens.dim() == 2:
            tokens = tokens[0]

        pitches = set()

        for token_id in tokens:
            token_str = tokenizer.id_to_token.get(token_id.item(), "")
            if token_str.startswith("NOTE_ON_"):
                pitch = int(token_str.split("_")[2])
                pitches.add(pitch)

        return len(pitches)

    @staticmethod
    def polyphony_rate(tokens: torch.Tensor, tokenizer) -> float:
        """
        Ratio of time with multiple simultaneous notes

        Args:
            tokens: Token tensor
            tokenizer: Tokenizer instance

        Returns:
            Polyphony rate (0-1)
        """
        if tokens.dim() == 2:
            tokens = tokens[0]

        active_notes = set()
        poly_time = 0
        total_time = 0
        current_time_ms = 0
        last_time_ms = 0

        for token_id in tokens:
            token_str = tokenizer.id_to_token.get(token_id.item(), "")

            if token_str.startswith("TIME_"):
                time_steps = int(token_str.split("_")[1])
                current_time_ms = time_steps * tokenizer.config.time_quantization_ms

                # Check if polyphonic at last time
                if len(active_notes) > 1 and current_time_ms > last_time_ms:
                    poly_time += current_time_ms - last_time_ms

                if current_time_ms > last_time_ms:
                    total_time += current_time_ms - last_time_ms

                last_time_ms = current_time_ms

            elif token_str.startswith("NOTE_ON_"):
                pitch = int(token_str.split("_")[2])
                active_notes.add(pitch)

            elif token_str.startswith("NOTE_OFF_"):
                pitch = int(token_str.split("_")[2])
                active_notes.discard(pitch)

        if total_time == 0:
            return 0.0

        return poly_time / total_time

    @staticmethod
    def rhythmic_entropy(tokens: torch.Tensor, tokenizer, num_bins: int = 32) -> float:
        """
        Entropy of IOI distribution (higher = more varied rhythm)

        Args:
            tokens: Token tensor
            tokenizer: Tokenizer instance
            num_bins: Number of bins for histogram

        Returns:
            Entropy value
        """
        if tokens.dim() == 2:
            tokens = tokens[0]

        iois = []
        onsets = []
        current_time_ms = 0

        for token_id in tokens:
            token_str = tokenizer.id_to_token.get(token_id.item(), "")

            if token_str.startswith("TIME_"):
                time_steps = int(token_str.split("_")[1])
                current_time_ms = time_steps * tokenizer.config.time_quantization_ms

            elif token_str.startswith("NOTE_ON_"):
                onsets.append(current_time_ms)

        if len(onsets) < 2:
            return 0.0

        iois = np.diff(onsets)

        # Histogram of IOIs
        hist, _ = np.histogram(iois, bins=num_bins)
        hist = hist / (hist.sum() + 1e-6)

        # Entropy
        ent = entropy(hist + 1e-10)

        return float(ent)

    @staticmethod
    def compute_all_metrics(
        tokens_gen: torch.Tensor,
        tokens_ref: torch.Tensor,
        tokenizer
    ) -> Dict[str, float]:
        """
        Compute all metrics at once

        Args:
            tokens_gen: Generated tokens
            tokens_ref: Reference tokens
            tokenizer: Tokenizer instance

        Returns:
            Dictionary of all metrics
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            metrics = {
                # Pitch class metrics
                'pitch_class_kl': MusicMetrics.pitch_class_kl_divergence(
                    tokens_gen, tokens_ref, tokenizer
                ),
                'pctm_similarity': MusicMetrics.pctm_cosine_similarity(
                    tokens_gen, tokens_ref, tokenizer
                ),

                # Note density
                'note_density_gen': MusicMetrics.note_density(tokens_gen, tokenizer),
                'note_density_ref': MusicMetrics.note_density(tokens_ref, tokenizer),
                'note_density_diff': abs(
                    MusicMetrics.note_density(tokens_gen, tokenizer) -
                    MusicMetrics.note_density(tokens_ref, tokenizer)
                ),

                # Inter-onset interval
                'avg_ioi_gen': MusicMetrics.average_ioi(tokens_gen, tokenizer),
                'avg_ioi_ref': MusicMetrics.average_ioi(tokens_ref, tokenizer),
                'avg_ioi_diff': abs(
                    MusicMetrics.average_ioi(tokens_gen, tokenizer) -
                    MusicMetrics.average_ioi(tokens_ref, tokenizer)
                ),

                # Pitch diversity
                'unique_pitches_gen': MusicMetrics.unique_pitches(tokens_gen, tokenizer),
                'unique_pitches_ref': MusicMetrics.unique_pitches(tokens_ref, tokenizer),

                # Polyphony
                'polyphony_rate_gen': MusicMetrics.polyphony_rate(tokens_gen, tokenizer),
                'polyphony_rate_ref': MusicMetrics.polyphony_rate(tokens_ref, tokenizer),

                # Rhythm
                'rhythmic_entropy_gen': MusicMetrics.rhythmic_entropy(tokens_gen, tokenizer),
                'rhythmic_entropy_ref': MusicMetrics.rhythmic_entropy(tokens_ref, tokenizer),
            }

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float]):
        """Pretty print metrics"""
        print("\n" + "="*60)
        print("Music Generation Metrics")
        print("="*60)

        print("\n📊 Pitch Class Metrics:")
        print(f"  KL Divergence:      {metrics['pitch_class_kl']:.4f} (lower is better)")
        print(f"  PCTM Similarity:    {metrics['pctm_similarity']:.4f} (higher is better)")

        print("\n🎹 Note Density:")
        print(f"  Generated:          {metrics['note_density_gen']:.2f} notes/5sec")
        print(f"  Reference:          {metrics['note_density_ref']:.2f} notes/5sec")
        print(f"  Difference:         {metrics['note_density_diff']:.2f}")

        print("\n⏱️  Inter-Onset Interval:")
        print(f"  Generated:          {metrics['avg_ioi_gen']:.3f} sec")
        print(f"  Reference:          {metrics['avg_ioi_ref']:.3f} sec")
        print(f"  Difference:         {metrics['avg_ioi_diff']:.3f} sec")

        print("\n🎵 Pitch Diversity:")
        print(f"  Generated:          {metrics['unique_pitches_gen']} unique pitches")
        print(f"  Reference:          {metrics['unique_pitches_ref']} unique pitches")

        print("\n🎼 Polyphony:")
        print(f"  Generated:          {metrics['polyphony_rate_gen']:.2%}")
        print(f"  Reference:          {metrics['polyphony_rate_ref']:.2%}")

        print("\n🥁 Rhythmic Complexity:")
        print(f"  Generated:          {metrics['rhythmic_entropy_gen']:.3f}")
        print(f"  Reference:          {metrics['rhythmic_entropy_ref']:.3f}")

        print("="*60 + "\n")


if __name__ == "__main__":
    # Test metrics
    from tatumflow import TatumFlowTokenizer

    print("Testing music metrics...")

    tokenizer = TatumFlowTokenizer()

    # Create dummy tokens
    tokens_gen = torch.randint(0, tokenizer.vocab_size, (1, 256))
    tokens_ref = torch.randint(0, tokenizer.vocab_size, (1, 256))

    # Compute metrics
    metrics = MusicMetrics.compute_all_metrics(tokens_gen, tokens_ref, tokenizer)

    # Print
    MusicMetrics.print_metrics(metrics)

    print("✅ Metrics module ready!")
