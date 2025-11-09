"""
QLoRA Integration with HuggingFace PEFT

Industry-standard parameter-efficient fine-tuning:
- Uses HuggingFace PEFT library (production standard)
- 4-bit quantization with bitsandbytes
- LoRA adapters for efficient training
- Compatible with HuggingFace Trainer

Why PEFT library?
- Industry standard (used by all major companies)
- Well-tested and maintained
- Easy integration with transformers
- Supports multiple PEFT methods (LoRA, AdaLoRA, IA3, etc.)

References:
- QLoRA paper: https://arxiv.org/abs/2305.14314
- PEFT library: https://github.com/huggingface/peft
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from transformers import BitsAndBytesConfig


@dataclass
class QLoRAConfig:
    """
    QLoRA configuration wrapper

    Simplifies QLoRA setup with sensible defaults
    """
    # LoRA parameters
    lora_rank: int = 8  # Rank of LoRA matrices (lower = fewer parameters)
    lora_alpha: int = 16  # LoRA scaling factor (usually 2x rank)
    lora_dropout: float = 0.1  # Dropout for LoRA layers

    # Target modules (which layers to apply LoRA)
    target_modules: Optional[List[str]] = None

    # Quantization
    load_in_4bit: bool = True  # Use 4-bit quantization
    bnb_4bit_compute_dtype: str = "float16"  # Compute dtype
    bnb_4bit_quant_type: str = "nf4"  # NormalFloat4 (better than standard 4-bit)
    bnb_4bit_use_double_quant: bool = True  # Double quantization for extra compression

    def __post_init__(self):
        if self.target_modules is None:
            # Default: apply LoRA to attention projections
            self.target_modules = [
                "query",  # Q projection
                "key",    # K projection
                "value",  # V projection
                "out_proj"  # Output projection
            ]


def get_bnb_config(qlora_config: QLoRAConfig) -> BitsAndBytesConfig:
    """
    Create BitsAndBytes quantization config

    BitsAndBytes is the industry standard for 4-bit quantization
    Developed by Tim Dettmers (creator of AdamW 8-bit)
    """
    return BitsAndBytesConfig(
        load_in_4bit=qlora_config.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, qlora_config.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=qlora_config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=qlora_config.bnb_4bit_use_double_quant
    )


def get_lora_config(qlora_config: QLoRAConfig) -> LoraConfig:
    """
    Create PEFT LoRA config

    LoRA (Low-Rank Adaptation) adds trainable low-rank matrices
    to frozen pretrained weights
    """
    return LoraConfig(
        r=qlora_config.lora_rank,
        lora_alpha=qlora_config.lora_alpha,
        lora_dropout=qlora_config.lora_dropout,
        target_modules=qlora_config.target_modules,
        bias="none",  # Don't add bias parameters
        task_type=TaskType.CAUSAL_LM  # For autoregressive generation
    )


def apply_qlora_to_model(
    model: nn.Module,
    qlora_config: Optional[QLoRAConfig] = None
) -> nn.Module:
    """
    Apply QLoRA to a model using HuggingFace PEFT

    This is the production-standard way to apply QLoRA

    Args:
        model: Base model (should be loaded with quantization)
        qlora_config: QLoRA configuration

    Returns:
        PEFT model with LoRA adapters

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>>
        >>> # Load model with 4-bit quantization
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "checkpoint",
        ...     quantization_config=get_bnb_config(qlora_config),
        ...     device_map="auto"
        ... )
        >>>
        >>> # Apply LoRA adapters
        >>> model = apply_qlora_to_model(model, qlora_config)
        >>>
        >>> # Now train!
        >>> trainer.train()
    """
    if qlora_config is None:
        qlora_config = QLoRAConfig()

    # Prepare model for k-bit training (gradient checkpointing, etc.)
    model = prepare_model_for_kbit_training(model)

    # Get LoRA config
    lora_config = get_lora_config(qlora_config)

    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model


def load_qlora_model(
    model_name_or_path: str,
    qlora_config: Optional[QLoRAConfig] = None,
    device_map: str = "auto"
):
    """
    Load a model with QLoRA configuration

    This is a convenience function for the common workflow:
    1. Load base model with 4-bit quantization
    2. Apply LoRA adapters
    3. Return PEFT model ready for training

    Args:
        model_name_or_path: Path to base model or HuggingFace model ID
        qlora_config: QLoRA configuration
        device_map: Device placement strategy

    Returns:
        PEFT model ready for QLoRA fine-tuning

    Example:
        >>> # Load base model with QLoRA
        >>> model = load_qlora_model(
        ...     "experiments/pretrained_music_transformer",
        ...     qlora_config=QLoRAConfig(lora_rank=8)
        ... )
        >>>
        >>> # Train with HuggingFace Trainer
        >>> trainer = Trainer(model=model, ...)
        >>> trainer.train()
    """
    from transformers import AutoModelForCausalLM

    if qlora_config is None:
        qlora_config = QLoRAConfig()

    # Create quantization config
    bnb_config = get_bnb_config(qlora_config)

    # Load model with 4-bit quantization
    print(f"Loading model from {model_name_or_path} with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True
    )

    # Apply LoRA adapters
    print("Applying LoRA adapters...")
    model = apply_qlora_to_model(model, qlora_config)

    print("✅ Model ready for QLoRA fine-tuning!")
    return model


def merge_and_save_qlora_model(
    peft_model,
    output_dir: str,
    save_merged: bool = True
):
    """
    Save QLoRA model

    Options:
    1. Save only LoRA adapters (tiny! ~10MB)
    2. Save merged model (base + adapters)

    In production, usually save adapters only and merge on-demand

    Args:
        peft_model: PEFT model with LoRA adapters
        output_dir: Output directory
        save_merged: Whether to save merged model

    Example:
        >>> # Option 1: Save adapters only (recommended)
        >>> peft_model.save_pretrained("checkpoints/lora_adapters")
        >>> # Size: ~10MB
        >>>
        >>> # Option 2: Save merged model
        >>> merge_and_save_qlora_model(
        ...     peft_model,
        ...     "checkpoints/merged_model",
        ...     save_merged=True
        ... )
        >>> # Size: ~600MB
    """
    import os

    # Save LoRA adapters (always)
    adapter_dir = os.path.join(output_dir, "adapters")
    peft_model.save_pretrained(adapter_dir)
    print(f"✅ LoRA adapters saved to {adapter_dir}")

    # Save merged model (optional)
    if save_merged:
        merged_dir = os.path.join(output_dir, "merged")

        # Merge LoRA weights into base model
        merged_model = peft_model.merge_and_unload()

        # Save merged model
        merged_model.save_pretrained(merged_dir)
        print(f"✅ Merged model saved to {merged_dir}")

    print(f"\n📁 Saved to {output_dir}/")
    print("   ├── adapters/      (LoRA weights, ~10MB)")
    if save_merged:
        print("   └── merged/        (Full model, ~600MB)")


def load_trained_qlora_model(
    base_model_path: str,
    adapter_path: str,
    device_map: str = "auto"
):
    """
    Load a trained QLoRA model for inference

    Args:
        base_model_path: Path to base model
        adapter_path: Path to LoRA adapters
        device_map: Device placement

    Returns:
        Model with LoRA adapters loaded

    Example:
        >>> # Load for inference
        >>> model = load_trained_qlora_model(
        ...     base_model_path="experiments/pretrained_music_transformer",
        ...     adapter_path="checkpoints/best_model/adapters"
        ... )
        >>>
        >>> # Generate
        >>> outputs = model.generate(input_ids, max_length=512)
    """
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    # Load base model
    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device_map,
        trust_remote_code=True
    )

    # Load LoRA adapters
    print(f"Loading LoRA adapters from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("✅ Model loaded and ready for inference!")
    return model


# Example usage for interviews / documentation
EXAMPLE_USAGE = """
# ============================================================
# QLoRA Fine-tuning Example (Production Standard)
# ============================================================

from models.qlora import (
    QLoRAConfig,
    load_qlora_model,
    merge_and_save_qlora_model
)
from transformers import Trainer, TrainingArguments

# 1. Configure QLoRA
qlora_config = QLoRAConfig(
    lora_rank=8,              # Low rank for efficiency
    lora_alpha=16,            # Scaling factor
    lora_dropout=0.1,         # Regularization
    target_modules=[          # Which layers to adapt
        "query", "key", "value", "out_proj"
    ],
    load_in_4bit=True,        # 4-bit quantization
    bnb_4bit_quant_type="nf4" # NormalFloat4 (best quality)
)

# 2. Load model with QLoRA
model = load_qlora_model(
    "experiments/music_transformer_base",
    qlora_config=qlora_config
)

# Model stats:
# - Total params: 150M
# - Trainable params: 2.8M (1.9%)
# - Memory: 8GB VRAM (vs 24GB for full fine-tune)

# 3. Configure training
training_args = TrainingArguments(
    output_dir="experiments/brad_mehldau_qlora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size: 16
    learning_rate=3e-4,              # Higher LR for LoRA
    num_train_epochs=5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    fp16=True,                       # Mixed precision
    report_to="wandb",               # Log to W&B
    load_best_model_at_end=True,
    metric_for_best_model="loss"
)

# 4. Create trainer (HuggingFace Trainer handles everything!)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# 5. Train!
trainer.train()

# 6. Save model
merge_and_save_qlora_model(
    model,
    "checkpoints/final_model",
    save_merged=True
)

# 7. Load for inference
from models.qlora import load_trained_qlora_model

model = load_trained_qlora_model(
    base_model_path="experiments/music_transformer_base",
    adapter_path="checkpoints/final_model/adapters"
)

# 8. Generate music!
outputs = model.generate(
    input_ids,
    max_length=1024,
    temperature=0.9,
    top_p=0.95
)
"""


if __name__ == "__main__":
    print("QLoRA Integration with HuggingFace PEFT\n")
    print("=" * 60)
    print("\nThis module provides production-standard QLoRA integration.")
    print("\nKey features:")
    print("  ✅ HuggingFace PEFT library (industry standard)")
    print("  ✅ 4-bit quantization with bitsandbytes")
    print("  ✅ Easy integration with Trainer")
    print("  ✅ Adapter-only saving (tiny checkpoints)")
    print("  ✅ Production-ready code")
    print("\n" + "=" * 60)
    print("\nExample usage:")
    print(EXAMPLE_USAGE)

    # Test configuration
    print("\n" + "=" * 60)
    print("Testing QLoRA configuration...\n")

    config = QLoRAConfig(
        lora_rank=8,
        lora_alpha=16,
        target_modules=["query", "key", "value"]
    )

    print(f"LoRA Config:")
    print(f"  Rank: {config.lora_rank}")
    print(f"  Alpha: {config.lora_alpha}")
    print(f"  Dropout: {config.lora_dropout}")
    print(f"  Target modules: {config.target_modules}")
    print(f"  Quantization: {'4-bit' if config.load_in_4bit else '8-bit'}")
    print(f"  Quant type: {config.bnb_4bit_quant_type}")

    print("\n✅ QLoRA configuration created successfully!")
    print("\nFor full example, see training/train.py")
