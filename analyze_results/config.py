"""Shared configuration constants for detective story analysis."""

# Model name mappings
model_dict = {
    'gemini-2.5-pro': 'Model_0', 'gemini-2.5-flash-tb0': 'Model_1', 'gemini-2.5-flash-tb-1': 'Model_2',
    'gemini-1.5-pro': 'Model_3', 'Llama-3.1-70B-Instruct': 'Model_4', 'Llama-3.3-70B-Instruct': 'Model_5',
    'gpt-4o': 'Model_6', 'gpt-4o-mini': 'Model_7', 'Llama-3.1-8B-Instruct': 'Model_8',
    'Llama-3.2-3B-Instruct': 'Model_9', 'sherlock': 'Real_0', 'poirot': 'Real_1',
    'gemini-3-flash-preview': 'Model_10', 'gemini-3.1-pro-preview': 'Model_11',
}
r_model_dict = {v: k for k, v in model_dict.items()}

# Analysis settings
NORMALIZE_NAIVE = False
NAIVE_THRESHOLD = 0.8
# Know-it-all "early reveal" analysis
EARLY_REVEAL_THRESHOLD = 0.5        # P(true culprit) must exceed this to count as "revealed"
EARLY_REVEAL_PARAGRAPHS = (1, 3, 5) # which paragraph numbers (from ids) to check
# USE_ARGMAX = False
USE_ARGMAX = True

REAL_STORY_S_MODEL = "gemini-1.5-flash"
# REAL_STORY_S_MODEL = "gemini-3-flash-preview"

ignore = ["sherlock 8"]

# Display names for plots
display_names = {
    'gemini-2.5-pro': 'Gemini 2.5 Pro',
    'gemini-2.5-flash': 'Gemini 2.5 Flash',
    'gemini-1.5-pro': 'Gemini 1.5 Pro',
    'gemini-1.5-flash': 'Gemini 1.5 Flash',
    'gemini-2.5-flash-tb0': 'Gemini 2.5 Flash TB0',
    'gemini-2.5-flash-tb-1': 'Gemini 2.5 Flash TB-1',
    'Llama-3.1-70B-Instruct': 'Llama 3.1 70B',
    'Llama-3.3-70B-Instruct': 'Llama 3.3 70B',
    'Llama-3.1-8B-Instruct': 'Llama 3.1 8B',
    'Llama-3.2-3B-Instruct': 'Llama 3.2 3B',
    'Llama-3.2-1B-Instruct': 'Llama 3.2 1B',
    'gpt-4o': 'GPT-4o',
    'gpt-4o-mini': 'GPT-4o Mini',
    'gemini-3-flash-preview': 'Gemini 3 Flash',
    'gemini-3.1-pro-preview': 'Gemini 3.1 Pro',
    'poirot': 'Poirot',
    'sherlock': 'Sherlock',
}

# Model lists for iteration
model_names_all = [
    'gemini-2.5-pro', 'gemini-2.5-flash-tb0', 'gemini-2.5-flash-tb-1',
    'gemini-1.5-pro', 'gemini-1.5-flash', 'Llama-3.1-70B-Instruct', 'Llama-3.3-70B-Instruct',
    'Llama-3.1-8B-Instruct', 'Llama-3.2-3B-Instruct', 'Llama-3.2-1B-Instruct',
    'gpt-4o', 'gpt-4o-mini'
]

model_names_with_human = [
    'gemini-2.5-pro', 'gemini-2.5-flash-tb0', 'gemini-2.5-flash-tb-1',
    'gemini-1.5-pro', 'Llama-3.1-70B-Instruct',
    'Llama-3.3-70B-Instruct', 'Llama-3.1-8B-Instruct',
    'Llama-3.2-3B-Instruct', 'gpt-4o', 'gpt-4o-mini'
]

# New models with partial data — included only when INCLUDE_NEW_MODELS is set in main.py
model_names_new = ['gemini-3-flash-preview', 'gemini-3.1-pro-preview']

# Experienced reader settings labels
# Settings 0-2 come from -es2 files:
#   0 = general instructions only (0-shot)
#   1 = 9 previous stories given
#   2 = 3 previous stories given
# Settings 3-5 come from -es3 files:
#   3 = different prompt, no previous stories (0-shot alt)
#   4 = 2 previous stories given
#   5 = 5 previous stories given
EXPERIENCED_READER_LABELS = {
    0: "Experienced Reader (0-shot)",
    1: "Experienced Reader (9-prev)",
    2: "Experienced Reader (3-prev)",
    3: "Experienced Reader (0-shot, alt prompt)",
    4: "Experienced Reader (2-prev)",
    5: "Experienced Reader (5-prev)",
}

# Short labels for plot legends
EXPERIENCED_READER_SHORT_LABELS = {
    0: "0-shot",
    1: "9-prev",
    2: "3-prev",
    3: "0-shot-alt",
    4: "2-prev",
    5: "5-prev",
}

# Colors for each experienced reader setting in plots
EXPERIENCED_READER_COLORS = {
    0: '#ffe599',   # yellow
    1: '#b6d7a8',   # green
    2: '#a4c2f4',   # blue
    3: '#f9cb9c',   # orange
    4: '#d9ead3',   # light green
    5: '#cfe2f3',   # light blue
}

# Ground-truth generation success rates and valid story counts (from paper table)
MODEL_GEN_SUCCESS = {
    'Llama-3.2-1B-Instruct': 0.27,
    'Llama-3.2-3B-Instruct': 0.73,
    'Llama-3.1-8B-Instruct': 0.90,
    'Llama-3.1-70B-Instruct': 0.83,
    'Llama-3.3-70B-Instruct': 1.00,
    'gemini-1.5-flash':       0.71,
    'gemini-1.5-pro':         1.00,
    'gemini-2.5-flash-tb0':   1.00,
    'gemini-2.5-flash-tb-1':  1.00,
    'gemini-2.5-pro':         1.00,
    'gemini-3-flash-preview':  1.00,
    'gemini-3.1-pro-preview':  1.00,
    'gpt-4o-mini':            0.57,
    'gpt-4o':                 0.67,
}

MODEL_N_VALID = {
    'Llama-3.2-1B-Instruct': 3,
    'Llama-3.2-3B-Instruct': 8,
    'Llama-3.1-8B-Instruct': 9,
    'Llama-3.1-70B-Instruct': 10,
    'Llama-3.3-70B-Instruct': 10,
    'gemini-1.5-flash':       9,
    'gemini-1.5-pro':         10,
    'gemini-2.5-flash-tb0':   10,
    'gemini-2.5-flash-tb-1':  10,
    'gemini-2.5-pro':         10,
    'gemini-3-flash-preview':  10,
    'gemini-3.1-pro-preview':  10,
    'gpt-4o-mini':            8,
    'gpt-4o':                 10,
}