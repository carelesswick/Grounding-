# Multi-word prompt descriptions for difficult/small categories.
CATEGORY_PROMPTS = {
    "coating_rusting": "rusting coating on steel bridge",
    "railing_rusting": "rusting railing on bridge",
    "nut_rusting": "rusting nut on steel structure",
    "coating_peeling_off": "peeling coating on bridge surface",
    "coating_dirty": "dirty coating on bridge surface",
    "nut_missing": "missing nut on steel structure",
    "nest": "bird nest on steel structure",
}

def prompt_for(label):
    return CATEGORY_PROMPTS.get(label, label)
