"""
Music Transformer Model - HuggingFace Compatible

Industry-standard implementation following HuggingFace conventions:
- PretrainedConfig for configuration
- PreTrainedModel base class
- Compatible with PEFT (LoRA/QLoRA)
- Compatible with HuggingFace Trainer
- Compatible with model hub upload/download

Architecture based on:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Music Transformer" (Huang et al., 2018) - Relative attention
- GPT-2 style decoder-only architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math
from dataclasses import dataclass

# HuggingFace imports
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


@dataclass
class MusicTransformerConfig(PretrainedConfig):
    """
    Configuration for Music Transformer

    Follows HuggingFace PretrainedConfig conventions for:
    - Automatic config saving/loading
    - Model hub integration
    - Easy hyperparameter management
    """

    model_type = "music_transformer"

    def __init__(
        self,
        vocab_size: int = 700,
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        # Music-specific
        use_relative_attention: bool = True,
        max_relative_position: int = 512,
        chord_vocab_size: int = 100,
        use_chord_conditioning: bool = True,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        # Music-specific
        self.use_relative_attention = use_relative_attention
        self.max_relative_position = max_relative_position
        self.chord_vocab_size = chord_vocab_size
        self.use_chord_conditioning = use_chord_conditioning


class RelativePositionBias(nn.Module):
    """
    Relative positional encoding from Music Transformer (Huang et al., 2018)

    Why relative positions?
    - Musical patterns repeat at different positions
    - Absolute positions don't generalize well for music
    - Relative positions capture musical structure better
    """

    def __init__(self, num_heads: int, max_relative_position: int = 512):
        super().__init__()
        self.num_heads = num_heads
        self.max_relative_position = max_relative_position

        # Learnable relative position embeddings
        self.relative_attention_bias = nn.Embedding(
            2 * max_relative_position + 1,
            num_heads
        )

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute relative position bias matrix

        Returns:
            bias: [num_heads, seq_len, seq_len]
        """
        # Position indices
        positions = torch.arange(seq_len, device=device)

        # Relative positions: i - j
        relative_positions = positions[None, :] - positions[:, None]

        # Clip to max distance
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_relative_position,
            self.max_relative_position
        )

        # Shift to positive range [0, 2*max+1)
        relative_positions += self.max_relative_position

        # Get embeddings
        bias = self.relative_attention_bias(relative_positions)  # [seq_len, seq_len, num_heads]
        bias = bias.permute(2, 0, 1)  # [num_heads, seq_len, seq_len]

        return bias


class MusicTransformerAttention(nn.Module):
    """
    Multi-head self-attention with relative positional encoding

    Follows HuggingFace conventions for PEFT compatibility
    """

    def __init__(self, config: MusicTransformerConfig):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Q, K, V projections (PEFT will target these)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Output projection
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Relative position bias
        if config.use_relative_attention:
            self.relative_bias = RelativePositionBias(
                config.num_attention_heads,
                config.max_relative_position
            )
        else:
            self.relative_bias = None

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-head attention"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len] - 1 for tokens to attend, 0 for masked
            output_attentions: Return attention weights

        Returns:
            outputs: [batch, seq_len, hidden_size]
            attention_weights: Optional[batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len = hidden_states.shape[:2]

        # Project Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Add relative position bias
        if self.relative_bias is not None:
            relative_bias = self.relative_bias(seq_len, hidden_states.device)
            attention_scores = attention_scores + relative_bias.unsqueeze(0)

        # Apply attention mask
        if attention_mask is not None:
            # Convert to attention mask: [batch, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores + attention_mask

        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Aggregate values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, seq_len, self.all_head_size)

        # Output projection
        outputs = self.out_proj(context_layer)

        return (outputs, attention_probs) if output_attentions else (outputs,)


class MusicTransformerBlock(nn.Module):
    """
    Transformer block with pre-layer normalization (GPT-2 style)

    Pre-LN is more stable for training and preferred in production
    """

    def __init__(self, config: MusicTransformerConfig):
        super().__init__()

        self.attention = MusicTransformerAttention(config)
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Pre-LN Transformer block

        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        hidden_states = residual + attn_outputs[0]

        # Feed-forward with residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        outputs = (hidden_states,) + attn_outputs[1:]  # Add attentions if output
        return outputs


class MusicTransformerModel(PreTrainedModel):
    """
    Base Music Transformer Model (without LM head)

    Extends PreTrainedModel for:
    - Automatic weight initialization
    - Model saving/loading
    - HuggingFace Hub integration
    - PEFT compatibility
    """

    config_class = MusicTransformerConfig
    base_model_prefix = "transformer"

    def __init__(self, config: MusicTransformerConfig):
        super().__init__(config)
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Positional embedding (absolute, for initial position awareness)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )

        # Chord conditioning (optional)
        if config.use_chord_conditioning:
            self.chord_embedding = nn.Embedding(
                config.chord_vocab_size,
                config.hidden_size
            )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            MusicTransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights
        self.post_init()

    def _init_weights(self, module):
        """Initialize weights (HuggingFace convention)"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chord_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            chord_ids: [batch, num_chords] - Optional chord conditioning
            position_ids: [batch, seq_len] - Optional position override
            output_attentions: Return attention weights
            output_hidden_states: Return all hidden states
            return_dict: Return ModelOutput object

        Returns:
            last_hidden_state: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Create attention mask if not provided (causal mask)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

        # Create causal mask (can only attend to past positions)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1
        )
        attention_mask = attention_mask.unsqueeze(1) & (~causal_mask)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)

        hidden_states = token_embeds + position_embeds

        # Add chord conditioning if provided
        if chord_ids is not None and self.config.use_chord_conditioning:
            chord_embeds = self.chord_embedding(chord_ids)
            # Average chord embeddings and add to sequence
            chord_embeds = chord_embeds.mean(dim=1, keepdim=True)
            hidden_states = hidden_states + chord_embeds.expand_as(hidden_states)

        hidden_states = self.dropout(hidden_states)

        # Pass through transformer blocks
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            block_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )
            hidden_states = block_outputs[0]

            if output_attentions:
                all_attentions += (block_outputs[1],)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return CausalLMOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )


class MusicTransformerForGeneration(PreTrainedModel):
    """
    Music Transformer with Language Modeling head

    For autoregressive music generation (predict next event)
    Compatible with HuggingFace generate() method
    """

    config_class = MusicTransformerConfig
    base_model_prefix = "transformer"

    def __init__(self, config: MusicTransformerConfig):
        super().__init__(config)
        self.transformer = MusicTransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights (optional, saves parameters)
        self.lm_head.weight = self.transformer.token_embedding.weight

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chord_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            chord_ids: [batch, num_chords]
            labels: [batch, seq_len] - For training

        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: Optional cross-entropy loss
        """
        # Forward through transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            chord_ids=chord_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        hidden_states = transformer_outputs.last_hidden_state

        # LM head
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        chord_ids: Optional[torch.Tensor] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive generation with top-p (nucleus) sampling

        Args:
            input_ids: [batch, start_len] - Start sequence
            max_length: Maximum sequence length
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (optional)
            chord_ids: Chord conditioning

        Returns:
            generated_ids: [batch, max_length]
        """
        pad_token_id = pad_token_id or self.config.pad_token_id
        eos_token_id = eos_token_id or self.config.eos_token_id

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Generation loop
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self(input_ids, chord_ids=chord_ids)
            logits = outputs.logits[:, -1, :]  # Last token logits

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if all sequences generated EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids


if __name__ == "__main__":
    """Test the model implementation"""
    print("Testing Music Transformer...")

    # Create config
    config = MusicTransformerConfig(
        vocab_size=700,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        max_position_embeddings=2048
    )

    print(f"Config: {config}")

    # Create model
    model = MusicTransformerForGeneration(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 700, (batch_size, seq_len))
    labels = torch.randint(0, 700, (batch_size, seq_len))

    outputs = model(input_ids, labels=labels)

    print(f"\nForward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {outputs.logits.shape}")
    print(f"  Loss: {outputs.loss.item():.4f}")

    # Test generation
    start_tokens = torch.randint(0, 700, (1, 10))
    generated = model.generate(start_tokens, max_length=50, temperature=0.9)

    print(f"\nGeneration:")
    print(f"  Start tokens: {start_tokens.shape}")
    print(f"  Generated: {generated.shape}")

    # Test config saving
    config.save_pretrained("test_config")
    print("\n✅ Config saved to test_config/")

    # Test model saving
    model.save_pretrained("test_model")
    print("✅ Model saved to test_model/")

    # Test loading
    loaded_model = MusicTransformerForGeneration.from_pretrained("test_model")
    print("✅ Model loaded from test_model/")

    print("\n✅ All tests passed!")
