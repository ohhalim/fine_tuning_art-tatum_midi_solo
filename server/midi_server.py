"""
MIDI Server for FL Studio Integration

Real-time communication with FL Studio via loopMIDI:
- Listens for chord progressions from FL Studio
- Generates Brad Mehldau solos
- Sends MIDI notes back to FL Studio
"""

import os
import sys
import threading
import queue
import time
from typing import List, Optional

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    print("⚠️  mido not installed. Install with: pip install mido python-rtmidi")
    MIDO_AVAILABLE = False

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.inference_server import BradMehldauGenerator, MIDIChordDetector


class MIDIServer:
    """
    MIDI Server for real-time Brad Mehldau generation

    Workflow:
    1. FL Studio sends chord progression via loopMIDI Port 2
    2. Server detects chords and buffers 4 chords
    3. Generates Brad Mehldau solo
    4. Sends MIDI notes to FL Studio via loopMIDI Port 1
    """

    def __init__(
        self,
        generator: BradMehldauGenerator,
        input_port_name: str = "loopMIDI Port 2",
        output_port_name: str = "loopMIDI Port 1",
        chord_buffer_size: int = 4
    ):
        """
        Args:
            generator: BradMehldauGenerator instance
            input_port_name: MIDI input port from FL Studio
            output_port_name: MIDI output port to FL Studio
            chord_buffer_size: Number of chords to buffer before generating
        """
        self.generator = generator
        self.input_port_name = input_port_name
        self.output_port_name = output_port_name
        self.chord_buffer_size = chord_buffer_size

        self.chord_detector = MIDIChordDetector()
        self.chord_queue = queue.Queue()
        self.running = False

        self.input_port = None
        self.output_port = None

    def start(self):
        """Start MIDI server"""
        if not MIDO_AVAILABLE:
            print("❌ mido not available. Cannot start MIDI server.")
            return

        self.running = True

        # Open MIDI ports
        try:
            print(f"🎹 Opening MIDI ports...")
            print(f"   Input:  {self.input_port_name}")
            print(f"   Output: {self.output_port_name}")

            self.input_port = mido.open_input(self.input_port_name)
            self.output_port = mido.open_output(self.output_port_name)

            print("   ✅ MIDI ports opened!")
        except Exception as e:
            print(f"❌ Error opening MIDI ports: {e}")
            print("\nTroubleshooting:")
            print("1. Install loopMIDI: https://www.tobias-erichsen.de/software/loopmidi.html")
            print("2. Create two virtual ports: 'loopMIDI Port 1' and 'loopMIDI Port 2'")
            print("3. Restart this script")
            return

        # Start threads
        listen_thread = threading.Thread(target=self._listen_chords, daemon=True)
        generate_thread = threading.Thread(target=self._generate_loop, daemon=True)

        listen_thread.start()
        generate_thread.start()

        print("\n" + "=" * 60)
        print("🎹 Brad Mehldau AI Generator - READY!")
        print("=" * 60)
        print("Listening for chords from FL Studio...")
        print("Play 4 chords to generate a solo!")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping server...")
            self.stop()

    def stop(self):
        """Stop MIDI server"""
        self.running = False

        if self.input_port:
            self.input_port.close()
        if self.output_port:
            self.output_port.close()

        print("✅ Server stopped")

    def _listen_chords(self):
        """Listen for chord input from FL Studio"""
        print("🎧 Chord listener started")

        current_chord = None
        last_chord_time = time.time()

        for msg in self.input_port:
            if not self.running:
                break

            # Note on
            if msg.type == 'note_on' and msg.velocity > 0:
                self.chord_detector.note_on(msg.note)

                # Detect chord
                detected = self.chord_detector.detect_chord()
                if detected and detected != current_chord:
                    current_chord = detected
                    last_chord_time = time.time()

                    print(f"🎵 Chord detected: {current_chord}")
                    self.chord_queue.put(current_chord)

            # Note off
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                self.chord_detector.note_off(msg.note)

                # If all notes released, prepare for next chord
                if not self.chord_detector.active_notes:
                    current_chord = None

            # Timeout: reset if no activity for 2 seconds
            if time.time() - last_chord_time > 2.0:
                self.chord_detector.clear()
                current_chord = None

    def _generate_loop(self):
        """Generation loop"""
        print("🎼 Generator loop started")

        while self.running:
            # Wait for enough chords
            if self.chord_queue.qsize() >= self.chord_buffer_size:
                # Collect chords
                chords = []
                for _ in range(self.chord_buffer_size):
                    try:
                        chord = self.chord_queue.get(timeout=0.1)
                        chords.append(chord)
                    except queue.Empty:
                        break

                if len(chords) >= self.chord_buffer_size:
                    print(f"\n{'=' * 60}")
                    print(f"🎹 Generating solo for: {' → '.join(chords)}")
                    print(f"{'=' * 60}")

                    # Generate
                    try:
                        piano_roll = self.generator.generate_solo(
                            chord_progression=chords,
                            num_steps=50,
                            guidance_scale=7.5,
                            temperature=0.8
                        )

                        # Convert to MIDI notes
                        notes = self.generator.piano_roll_to_midi_notes(piano_roll)

                        # Send to FL Studio
                        self._send_midi_notes(notes)

                        print(f"✅ Sent {len(notes)} notes to FL Studio\n")

                    except Exception as e:
                        print(f"❌ Generation error: {e}")

            time.sleep(0.1)

    def _send_midi_notes(self, notes: List[dict], time_scale: float = 0.05):
        """
        Send MIDI notes to FL Studio

        Args:
            notes: List of {'pitch', 'start', 'end', 'velocity'}
            time_scale: Time scale (seconds per time step)
        """
        if not notes:
            return

        # Sort by start time
        notes = sorted(notes, key=lambda n: n['start'])

        # Send notes
        for note in notes:
            # Wait until note start time
            time.sleep(note['start'] * time_scale)

            # Note ON
            self.output_port.send(mido.Message(
                'note_on',
                note=note['pitch'],
                velocity=note['velocity']
            ))

            # Calculate duration
            duration = (note['end'] - note['start']) * time_scale

            # Note OFF (in separate thread to allow polyphony)
            def send_note_off(pitch, delay):
                time.sleep(delay)
                self.output_port.send(mido.Message(
                    'note_off',
                    note=pitch,
                    velocity=0
                ))

            threading.Thread(
                target=send_note_off,
                args=(note['pitch'], duration),
                daemon=True
            ).start()


def list_midi_ports():
    """List available MIDI ports"""
    if not MIDO_AVAILABLE:
        print("❌ mido not available")
        return

    print("Available MIDI Input Ports:")
    for port in mido.get_input_names():
        print(f"  - {port}")

    print("\nAvailable MIDI Output Ports:")
    for port in mido.get_output_names():
        print(f"  - {port}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Brad Mehldau MIDI Server")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda", help="Device (cuda/mps/cpu)")
    parser.add_argument("--input_port", default="loopMIDI Port 2", help="MIDI input port")
    parser.add_argument("--output_port", default="loopMIDI Port 1", help="MIDI output port")
    parser.add_argument("--chord_buffer", type=int, default=4, help="Chord buffer size")
    parser.add_argument("--list_ports", action="store_true", help="List MIDI ports and exit")

    args = parser.parse_args()

    if args.list_ports:
        list_midi_ports()
        return

    # Load generator
    print("Loading Brad Mehldau Generator...")
    generator = BradMehldauGenerator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        quantize=True
    )

    # Start server
    server = MIDIServer(
        generator=generator,
        input_port_name=args.input_port,
        output_port_name=args.output_port,
        chord_buffer_size=args.chord_buffer
    )

    server.start()


if __name__ == "__main__":
    # If no args, show help
    if len(sys.argv) == 1:
        print("=" * 60)
        print("Brad Mehldau AI Generator - MIDI Server")
        print("=" * 60)
        print("\nUsage:")
        print("  python server/midi_server.py --checkpoint ./checkpoints/brad_final/best.pt")
        print("\nOptions:")
        print("  --list_ports      List available MIDI ports")
        print("  --device DEVICE   Device (cuda/mps/cpu)")
        print("  --input_port      Input MIDI port (default: loopMIDI Port 2)")
        print("  --output_port     Output MIDI port (default: loopMIDI Port 1)")
        print("\nFirst time setup:")
        print("1. Install loopMIDI: https://www.tobias-erichsen.de/software/loopmidi.html")
        print("2. Create two virtual ports: 'loopMIDI Port 1' and 'loopMIDI Port 2'")
        print("3. Run this script with --checkpoint argument")
    else:
        main()
