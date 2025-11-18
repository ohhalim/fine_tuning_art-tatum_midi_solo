"""
TatumFlow: Hierarchical Latent Diffusion for Jazz Improvisation

A state-of-the-art model for generating expressive jazz piano improvisations
with explicit control over musical style, structure, and creativity.

Key Features:
- Multi-scale temporal modeling (note/beat/phrase levels)
- Latent diffusion in symbolic domain
- Explicit music theory disentanglement (harmony, melody, rhythm, dynamics)
- Bidirectional context modeling
- Style VAE for controllable generation
- Multiple generation modes: continuation, style transfer, improvisation

Example usage:

    from tatumflow import TatumFlow, TatumFlowTokenizer, TatumFlowGenerator
    from tatumflow import load_model_from_checkpoint

    # Load model
    model, tokenizer = load_model_from_checkpoint('checkpoints/best.pt')

    # Create generator
    generator = TatumFlowGenerator(model, tokenizer)

    # Generate continuation
    generated = generator.generate_continuation(
        prompt_midi='input.mid',
        num_tokens=512,
        temperature=1.0
    )

    # Save output
    generator.tokens_to_midi(generated, 'output.mid')
"""

__version__ = '1.0.0'
__author__ = 'TatumFlow Team'

from .model import (
    TatumFlow,
    create_tatumflow_model,
    MusicTheoryEncoder,
    LatentDiffusionCore,
    MultiScaleAttention
)

from .tokenizer import (
    TatumFlowTokenizer,
    TokenizerConfig
)

from .dataset import (
    MIDIDataset,
    ArtTatumDataset,
    create_dataloaders
)

from .train import (
    TatumFlowTrainer,
    TatumFlowLoss
)

from .generate import (
    TatumFlowGenerator,
    load_model_from_checkpoint
)

from .utils import (
    set_seed,
    load_config,
    count_parameters,
    get_device,
    print_model_summary
)

from .metrics import MusicMetrics

from .train_amp import (
    TatumFlowTrainerAMP,
    TatumFlowTrainerAMPEMA,
    EMA
)

__all__ = [
    # Model
    'TatumFlow',
    'create_tatumflow_model',
    'MusicTheoryEncoder',
    'LatentDiffusionCore',
    'MultiScaleAttention',

    # Tokenizer
    'TatumFlowTokenizer',
    'TokenizerConfig',

    # Dataset
    'MIDIDataset',
    'ArtTatumDataset',
    'create_dataloaders',

    # Training
    'TatumFlowTrainer',
    'TatumFlowLoss',
    'TatumFlowTrainerAMP',
    'TatumFlowTrainerAMPEMA',
    'EMA',

    # Generation
    'TatumFlowGenerator',
    'load_model_from_checkpoint',

    # Metrics
    'MusicMetrics',

    # Utils
    'set_seed',
    'load_config',
    'count_parameters',
    'get_device',
    'print_model_summary',
]
