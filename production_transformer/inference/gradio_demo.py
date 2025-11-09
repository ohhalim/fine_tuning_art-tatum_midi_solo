"""
Gradio Demo for Music Transformer

Interactive web interface for music generation
Perfect for portfolio and job interviews!

Gradio is used by:
- HuggingFace (all their demo apps)
- Stability AI (Stable Diffusion demos)
- Many startups for quick prototypes
"""

import gradio as gr
import os
import sys
from pathlib import Path
from typing import Tuple, Optional
import tempfile

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.generator import MusicGenerator


class GradioMusicDemo:
    """
    Gradio demo wrapper for Music Generator
    """

    def __init__(self, checkpoint_path: str):
        """
        Initialize demo

        Args:
            checkpoint_path: Path to trained model
        """
        print("Loading model...")
        self.generator = MusicGenerator(checkpoint_path)
        print("✅ Model loaded!\n")

    def generate_music(
        self,
        chord_progression: str,
        max_length: int,
        temperature: float,
        top_p: float,
        num_samples: int
    ) -> Tuple[str, str]:
        """
        Generate music from chord progression

        Args:
            chord_progression: Space-separated chords
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            num_samples: Number of samples to generate

        Returns:
            Tuple of (info_text, midi_file_path)
        """
        try:
            # Generate
            generated = self.generator.generate(
                prompt=chord_progression if chord_progression.strip() else None,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_samples
            )

            # Save to temporary file
            # (In production, you'd save to a permanent location)
            temp_dir = tempfile.mkdtemp()
            midi_paths = []

            for i, tokens in enumerate(generated):
                output_path = os.path.join(temp_dir, f"generated_{i+1}.mid")
                midi = self.generator.tokens_to_midi(tokens, output_path)
                midi_paths.append(output_path)

                # Get info
                duration = midi.get_end_time()
                num_notes = sum(len(inst.notes) for inst in midi.instruments)

                info = f"✅ Generated {num_samples} sample(s)\n\n"
                info += f"Sample {i+1}:\n"
                info += f"  Duration: {duration:.2f}s\n"
                info += f"  Notes: {num_notes}\n"
                info += f"  Tokens: {len(tokens)}\n"

            # Return first MIDI file
            return info, midi_paths[0]

        except Exception as e:
            error_msg = f"❌ Error generating music: {str(e)}"
            return error_msg, None

    def create_interface(self) -> gr.Blocks:
        """
        Create Gradio interface

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Brad Mehldau AI Generator") as demo:
            gr.Markdown("""
            # 🎹 Brad Mehldau AI Piano Generator

            Generate Brad Mehldau-style jazz piano solos using a fine-tuned Music Transformer.

            ## How to use:
            1. Enter a chord progression (e.g., "Cmaj7 Am7 Dm7 G7")
            2. Adjust generation parameters
            3. Click "Generate"
            4. Download and play the MIDI file!

            ## Available chords:
            - Major: Cmaj, Cmaj7, Cmaj9, Cmaj11, Cmaj13
            - Minor: Cmin, Cmin7, Cmin9, Cmin11, Cmin13
            - Dominant: C7, C9, C11, C13
            - Altered: Cdim, Caug, Cm7b5
            - Roots: C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B

            ---
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    # Input section
                    gr.Markdown("### 🎵 Input")

                    chord_input = gr.Textbox(
                        label="Chord Progression",
                        placeholder="Cmaj7 Am7 Dm7 G7",
                        value="Cmaj7 Am7 Dm7 G7",
                        info="Space-separated chords (leave empty for free generation)"
                    )

                    gr.Markdown("### ⚙️ Generation Parameters")

                    max_length = gr.Slider(
                        minimum=128,
                        maximum=2048,
                        value=512,
                        step=128,
                        label="Max Length",
                        info="Longer = more music (but slower)"
                    )

                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.9,
                        step=0.1,
                        label="Temperature",
                        info="Higher = more random/creative"
                    )

                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        label="Top-p (Nucleus Sampling)",
                        info="Lower = more focused"
                    )

                    num_samples = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=1,
                        step=1,
                        label="Number of Samples",
                        info="Generate multiple variations"
                    )

                    generate_btn = gr.Button("🎹 Generate Music", variant="primary")

                with gr.Column(scale=1):
                    # Output section
                    gr.Markdown("### 🎼 Output")

                    info_output = gr.Textbox(
                        label="Generation Info",
                        lines=10,
                        interactive=False
                    )

                    midi_output = gr.File(
                        label="Generated MIDI",
                        file_types=[".mid"]
                    )

                    gr.Markdown("""
                    ### 💡 Tips for Best Results:

                    1. **Chord progressions**: Use standard jazz progressions (ii-V-I, etc.)
                    2. **Temperature**: 0.8-1.0 for realistic, 1.2-1.5 for experimental
                    3. **Length**: 512-1024 for a complete solo
                    4. **Free generation**: Leave chords empty for free improvisation

                    ### 📊 Model Info:
                    - Architecture: Music Transformer with Relative Attention
                    - Training: QLoRA fine-tuning on Brad Mehldau performances
                    - Parameters: 150M total, 2.8M trainable (1.9%)
                    - Dataset: ~50 hours of Brad Mehldau MIDI
                    """)

            # Connect button to function
            generate_btn.click(
                fn=self.generate_music,
                inputs=[chord_input, max_length, temperature, top_p, num_samples],
                outputs=[info_output, midi_output]
            )

            # Example inputs
            gr.Examples(
                examples=[
                    ["Cmaj7 Am7 Dm7 G7", 512, 0.9, 0.95, 1],  # ii-V-I in C
                    ["Dm7 G7 Cmaj7 Am7", 512, 0.9, 0.95, 1],  # Classic progression
                    ["Fmaj7 Bbmaj7 Ebmaj7 Abmaj7", 768, 1.0, 0.9, 1],  # Modal
                    ["", 512, 1.2, 0.9, 1],  # Free improvisation
                ],
                inputs=[chord_input, max_length, temperature, top_p, num_samples],
                label="Example Prompts"
            )

        return demo


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Launch Gradio demo")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run demo on")
    parser.add_argument("--share", action="store_true",
                       help="Create public share link")
    parser.add_argument("--server_name", type=str, default="0.0.0.0",
                       help="Server name (0.0.0.0 for all interfaces)")

    args = parser.parse_args()

    # Create demo
    demo_wrapper = GradioMusicDemo(args.checkpoint)
    demo = demo_wrapper.create_interface()

    # Launch
    print("\n" + "=" * 60)
    print("🎹 Launching Gradio demo...")
    print("=" * 60 + "\n")

    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
