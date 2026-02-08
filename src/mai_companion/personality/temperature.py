"""Trait-to-temperature mapping.

Maps personality trait values to an LLM sampling temperature.  The formula
is data-driven: ``TEMPERATURE_ADJUSTMENTS`` defines which traits influence
temperature and by how much.  Adding a new trait in a later wave means
adding one entry to the dict.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Temperature adjustments per trait
# ---------------------------------------------------------------------------

# Maps trait name (str) to a weight.  Positive = higher trait value raises
# temperature (more creative/varied output).  Negative = higher trait value
# lowers temperature (more focused/measured output).
#
# Wave 1 adjustments only.  Wave 2/3 traits add entries here when implemented.
TEMPERATURE_ADJUSTMENTS: dict[str, float] = {
    "humor": +0.15,       # more humor -> more creative/unexpected
    "directness": -0.07,  # more direct -> more focused
    "patience": -0.05,    # more patient -> more measured
    "laziness": -0.03,    # more lazy -> slightly less creative effort
    # Wave 2 (added later):
    # "curiosity": +0.08,
    # "emotional_depth": +0.07,
    # Wave 3 (added later):
    # "special_speech": +0.05,
}

_BASE_TEMPERATURE = 0.7
_MIN_TEMPERATURE = 0.3
_MAX_TEMPERATURE = 1.4


def compute_temperature(traits: dict[str, float]) -> float:
    """Compute LLM temperature from personality traits.

    The formula starts from a base of 0.7 and adjusts based on
    trait values.  Each trait's contribution is:

        delta = weight * (trait_value - 0.5) * 2.0

    This means a trait at 0.5 (medium) has zero effect.  A trait at 0.9
    contributes ``weight * 0.8``, and at 0.1 contributes ``weight * -0.8``.

    Parameters
    ----------
    traits:
        Dict of trait name to float value (0.0-1.0).

    Returns
    -------
    float
        Clamped temperature in [0.3, 1.4].
    """
    temp = _BASE_TEMPERATURE

    for trait_name, weight in TEMPERATURE_ADJUSTMENTS.items():
        if trait_name in traits:
            delta = weight * (traits[trait_name] - 0.5) * 2.0
            temp += delta

    return max(_MIN_TEMPERATURE, min(_MAX_TEMPERATURE, round(temp, 3)))


def describe_temperature(temp: float) -> str:
    """Return a human-readable description of a temperature value.

    Useful for debugging and logging.

    Parameters
    ----------
    temp:
        The temperature value.

    Returns
    -------
    str
        Description like "0.85 -- moderately creative".
    """
    if temp <= 0.3:
        desc = "very focused/deterministic"
    elif temp <= 0.5:
        desc = "focused"
    elif temp <= 0.7:
        desc = "balanced"
    elif temp <= 0.9:
        desc = "moderately creative"
    elif temp <= 1.1:
        desc = "creative"
    elif temp <= 1.3:
        desc = "very creative"
    else:
        desc = "highly creative/unpredictable"

    return f"{temp:.2f} -- {desc}"
