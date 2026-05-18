import torch

from third_party.midi_processor.processor import RANGE_NOTE_ON, RANGE_NOTE_OFF, RANGE_VEL, RANGE_TIME_SHIFT

SEPERATOR               = "========================="

# Taken from the paper
ADAM_BETA_1             = 0.9
ADAM_BETA_2             = 0.98
ADAM_EPSILON            = 10e-9

LR_DEFAULT_START        = 1.0
SCHEDULER_WARMUP_STEPS  = 4000
# LABEL_SMOOTHING_E       = 0.1

# DROPOUT_P               = 0.1

TOKEN_END               = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
TOKEN_PAD               = TOKEN_END + 1

TOKEN_COND_SEP          = TOKEN_PAD + 1
TOKEN_BAR               = TOKEN_PAD + 2
TOKEN_ROLE_LEAD         = TOKEN_PAD + 3
TOKEN_TEMPO_SLOW        = TOKEN_PAD + 4
TOKEN_TEMPO_MEDIUM      = TOKEN_PAD + 5
TOKEN_TEMPO_DANCE       = TOKEN_PAD + 6
TOKEN_TEMPO_FAST        = TOKEN_PAD + 7

CONTROL_TOKEN_NAMES     = {
    TOKEN_COND_SEP: "COND_SEP",
    TOKEN_BAR: "BAR",
    TOKEN_ROLE_LEAD: "ROLE_LEAD",
    TOKEN_TEMPO_SLOW: "TEMPO_SLOW",
    TOKEN_TEMPO_MEDIUM: "TEMPO_MEDIUM",
    TOKEN_TEMPO_DANCE: "TEMPO_DANCE",
    TOKEN_TEMPO_FAST: "TEMPO_FAST",
}

CONTROL_TOKEN_IDS       = {name: token for token, name in CONTROL_TOKEN_NAMES.items()}

TOKEN_CONTROL_START     = TOKEN_COND_SEP
TOKEN_CONTROL_END       = TOKEN_TEMPO_FAST

VOCAB_SIZE              = TOKEN_CONTROL_END + 1

TORCH_FLOAT             = torch.float32
TORCH_INT               = torch.int32

TORCH_LABEL_TYPE        = torch.long

PREPEND_ZEROS_WIDTH     = 4
