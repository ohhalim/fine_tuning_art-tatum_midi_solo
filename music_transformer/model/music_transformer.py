import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import random

from utilities.constants import *
from utilities.device import get_device

from .positional_encoding import PositionalEncoding
from .rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR


def _sample_probs_from_logits(logits, temperature=1.0, top_k=None, top_p=None):
    temperature = max(float(temperature or 1.0), 1e-6)
    probs = torch.softmax(logits / temperature, dim=-1)

    if top_k is not None and int(top_k) > 0:
        k = min(int(top_k), probs.shape[-1])
        threshold = torch.topk(probs, k, dim=-1).values[..., -1, None]
        probs = probs.masked_fill(probs < threshold, 0.0)

    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > float(top_p)
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(probs, dtype=torch.bool)
        remove.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
        probs = probs.masked_fill(remove, 0.0)

    total = probs.sum(dim=-1, keepdim=True)
    if torch.any(total <= 0):
        return torch.softmax(logits, dim=-1)
    return probs / total


def _grammar_update_active_pitches(active_pitches: set, token: int) -> None:
    if 0 <= token < RANGE_NOTE_ON:
        active_pitches.add(token)
    elif RANGE_NOTE_ON <= token < RANGE_NOTE_ON + RANGE_NOTE_OFF:
        active_pitches.discard(token - RANGE_NOTE_ON)


def _apply_grammar_mask(token_logits, active_pitches: set):
    # Block tokens that decode_midi would discard: note_off without an active
    # note_on (orphan), and note_on for an already-active pitch (silently
    # overwrites the pending note_on in _merge_note).
    vocab = token_logits.shape[-1]
    off_start = RANGE_NOTE_ON
    off_end = min(RANGE_NOTE_ON + RANGE_NOTE_OFF, vocab)
    mask = torch.zeros(vocab, dtype=torch.bool, device=token_logits.device)
    if off_start < vocab:
        mask[off_start:off_end] = True
    for pitch in active_pitches:
        off_idx = RANGE_NOTE_ON + pitch
        if off_idx < vocab:
            mask[off_idx] = False
        if pitch < min(RANGE_NOTE_ON, vocab):
            mask[pitch] = True
    return token_logits.masked_fill(mask, float("-inf"))


# MusicTransformer
class MusicTransformer(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Music Transformer reproduction from https://arxiv.org/abs/1809.04281. Arguments allow for
    tweaking the transformer architecture (https://arxiv.org/abs/1706.03762) and the rpr argument
    toggles Relative Position Representations (RPR - https://arxiv.org/abs/1803.02155).

    Supports training and generation using Pytorch's nn.Transformer class with dummy decoder to
    make a decoder-only transformer architecture

    For RPR support, there is modified Pytorch 1.2.0 code in rpr.py. Modified source will be
    kept up to date with Pytorch revisions only as necessary.
    ----------
    """

    def __init__(self, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence=2048, rpr=False):
        super(MusicTransformer, self).__init__()

        self.dummy      = DummyDecoder()

        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.max_seq    = max_sequence
        self.rpr        = rpr

        # Input embedding
        self.embedding = nn.Embedding(VOCAB_SIZE, self.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)

        # Base transformer
        if(not self.rpr):
            # To make a decoder-only transformer we need to use masked encoder layers
            # Dummy decoder to essentially just return the encoder output
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy
            )
        # RPR Transformer
        else:
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)
            encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_norm)
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy, custom_encoder=encoder
            )

        # Final output is a softmaxed linear layer
        self.Wout       = nn.Linear(self.d_model, VOCAB_SIZE)
        self.softmax    = nn.Softmax(dim=-1)

    # forward
    def forward(self, x, mask=True):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Takes an input sequence and outputs predictions using a sequence to sequence method.

        A prediction at one index is the "next" prediction given all information seen previously.
        ----------
        """

        if(mask is True):
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(get_device())
        else:
            mask = None

        x = self.embedding(x)

        # Input shape is (max_seq, batch_size, d_model)
        x = x.permute(1,0,2)

        x = self.positional_encoding(x)

        # Since there are no true decoder layers, the tgt is unused
        # Pytorch wants src and tgt to have some equal dims however
        x_out = self.transformer(src=x, tgt=x, src_mask=mask)

        # Back to (batch_size, max_seq, d_model)
        x_out = x_out.permute(1,0,2)

        y = self.Wout(x_out)
        # y = self.softmax(y)

        del mask

        # They are trained to predict the next note in sequence (we don't need the last one)
        return y

    # generate
    def generate(
        self,
        primer=None,
        target_seq_length=1024,
        beam=0,
        beam_chance=1.0,
        temperature=1.0,
        top_k=None,
        top_p=None,
        sample_vocab_size=None,
        grammar_mask=False,
    ):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Generates midi given a primer sample. Music can be generated using a probability distribution over
        the softmax probabilities (recommended) or by using a beam search.
        ----------
        """

        assert (not self.training), "Cannot generate while in training mode"
        sample_vocab_size = int(sample_vocab_size or TOKEN_END)
        if sample_vocab_size <= 0 or sample_vocab_size > VOCAB_SIZE:
            raise ValueError(f"sample_vocab_size out of range: {sample_vocab_size}")

        print("Generating sequence of max length:", target_seq_length)

        gen_seq = torch.full((1,target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())

        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())


        # print("primer:",primer)
        # print(gen_seq)
        active_pitches: set = set()
        if grammar_mask:
            for tok in primer.flatten().tolist():
                _grammar_update_active_pitches(active_pitches, int(tok))

        cur_i = num_primer
        while(cur_i < target_seq_length):
            # gen_seq_batch     = gen_seq.clone()
            logits = self.forward(gen_seq[..., :cur_i])[..., :sample_vocab_size]
            token_logits = logits[:, cur_i - 1, :]
            if grammar_mask:
                token_logits = _apply_grammar_mask(token_logits, active_pitches)
            token_probs = _sample_probs_from_logits(
                token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if(beam == 0):
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0,1)

            if(beam_ran <= beam_chance):
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)

                beam_rows = top_i // VOCAB_SIZE
                beam_cols = top_i % sample_vocab_size

                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols

            else:
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                # print("next token:",next_token)
                gen_seq[:, cur_i] = next_token
                if grammar_mask:
                    _grammar_update_active_pitches(active_pitches, int(next_token))


                # Let the transformer decide to end if it wants to
                if(next_token == TOKEN_END):
                    print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                    break

            cur_i += 1
            if(cur_i % 50 == 0):
                print(cur_i, "/", target_seq_length)

        return gen_seq[:, :cur_i]

# Used as a dummy to nn.Transformer
# DummyDecoder
class DummyDecoder(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    A dummy decoder that returns its input. Used to make the Pytorch transformer into a decoder-only
    architecture (stacked encoders with dummy decoder fits the bill)
    ----------
    """

    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, **kwargs):
        """
        ----------
        Author: Damon Gwinn
        Modified: Added **kwargs for PyTorch 2.x compatibility
        ----------
        Returns the input (memory)
        ----------
        """

        return memory
