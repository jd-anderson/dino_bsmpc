import numpy as np
from typing import Dict, List, Tuple, Optional

TEST_BACKGROUNDS = {
    'slight_change': {
        'background_builtin': 'checker',
        'background_rgb1': '0.4 0.5 0.6',
        'background_rgb2': '0.3 0.4 0.5'
    },
    'gradient': {
        'background_builtin': 'gradient',
        'background_rgb1': '0.2 0.3 0.4',
        'background_rgb2': '0.1 0.2 0.3'
    },
    'gradient_aggressive': {
        'background_builtin': 'gradient',
        'background_rgb1': '0.18 0.05 0.35',
        'background_rgb2': '0.5 0.22 0.55'
    }
}

TEST_DISTRACTOR_RGBA = (0.0, 0.0, 0.0, 1.0)

CANONICAL_BACKGROUND = {
    'background_builtin': 'checker',
    'background_rgb1': '0.2 0.3 0.4',
    'background_rgb2': '0.1 0.2 0.3'
}

# ========================
# DOMAIN RANDOMIZATION SET
# ========================

DR_BACKGROUNDS = [
    # checker variations
    {
        'background_builtin': 'checker',
        'background_rgb1': '0.5 0.3 0.2',
        'background_rgb2': '0.3 0.2 0.1'
    },
    {
        'background_builtin': 'checker',
        'background_rgb1': '0.3 0.4 0.5',
        'background_rgb2': '0.2 0.3 0.4'
    },
    {
        'background_builtin': 'checker',
        'background_rgb1': '0.6 0.4 0.3',
        'background_rgb2': '0.4 0.3 0.2'
    },
    {
        'background_builtin': 'checker',
        'background_rgb1': '0.25 0.35 0.45',
        'background_rgb2': '0.15 0.25 0.35'
    },
    {
        'background_builtin': 'checker',
        'background_rgb1': '0.35 0.25 0.45',
        'background_rgb2': '0.25 0.15 0.35'
    },
    {
        'background_builtin': 'checker',
        'background_rgb1': '0.45 0.35 0.25',
        'background_rgb2': '0.35 0.25 0.15'
    },
    {
        'background_builtin': 'checker',
        'background_rgb1': '0.4 0.45 0.5',
        'background_rgb2': '0.3 0.35 0.4'
    },
    {
        'background_builtin': 'checker',
        'background_rgb1': '0.55 0.35 0.4',
        'background_rgb2': '0.45 0.25 0.3'
    },
    {
        'background_builtin': 'checker',
        'background_rgb1': '0.3 0.5 0.4',
        'background_rgb2': '0.2 0.4 0.3'
    },
    {
        'background_builtin': 'checker',
        'background_rgb1': '0.5 0.4 0.5',
        'background_rgb2': '0.4 0.3 0.4'
    },
    {
        'background_builtin': 'checker',
        'background_rgb1': '0.4 0.3 0.5',
        'background_rgb2': '0.3 0.2 0.4'
    },
    {
        'background_builtin': 'checker',
        'background_rgb1': '0.5 0.5 0.4',
        'background_rgb2': '0.4 0.4 0.3'
    },
    # gradient variations
    {
        'background_builtin': 'gradient',
        'background_rgb1': '0.3 0.2 0.4',
        'background_rgb2': '0.2 0.1 0.3'
    },
    {
        'background_builtin': 'gradient',
        'background_rgb1': '0.4 0.3 0.5',
        'background_rgb2': '0.3 0.2 0.4'
    },
    {
        'background_builtin': 'gradient',
        'background_rgb1': '0.25 0.15 0.35',
        'background_rgb2': '0.15 0.1 0.25'
    },
    {
        'background_builtin': 'gradient',
        'background_rgb1': '0.35 0.2 0.3',
        'background_rgb2': '0.25 0.1 0.2'
    },
    {
        'background_builtin': 'gradient',
        'background_rgb1': '0.3 0.35 0.4',
        'background_rgb2': '0.2 0.25 0.3'
    },
    {
        'background_builtin': 'gradient',
        'background_rgb1': '0.45 0.3 0.35',
        'background_rgb2': '0.35 0.2 0.25'
    },
    {
        'background_builtin': 'gradient',
        'background_rgb1': '0.4 0.4 0.5',
        'background_rgb2': '0.3 0.3 0.4'
    },
    {
        'background_builtin': 'gradient',
        'background_rgb1': '0.35 0.45 0.4',
        'background_rgb2': '0.25 0.35 0.3'
    },
    {
        'background_builtin': 'gradient',
        'background_rgb1': '0.5 0.35 0.4',
        'background_rgb2': '0.4 0.25 0.3'
    },
    {
        'background_builtin': 'gradient',
        'background_rgb1': '0.3 0.4 0.5',
        'background_rgb2': '0.2 0.3 0.4'
    },
    {
        'background_builtin': 'gradient',
        'background_rgb1': '0.4 0.25 0.35',
        'background_rgb2': '0.3 0.15 0.25'
    },
    {
        'background_builtin': 'gradient',
        'background_rgb1': '0.5 0.4 0.45',
        'background_rgb2': '0.4 0.3 0.35'
    },
    {
        'background_builtin': 'gradient',
        'background_rgb1': '0.35 0.3 0.45',
        'background_rgb2': '0.25 0.2 0.35'
    },
    {
        'background_builtin': 'gradient',
        'background_rgb1': '0.45 0.4 0.35',
        'background_rgb2': '0.35 0.3 0.25'
    },
    # flat variations
    {
        'background_builtin': 'flat',
        'background_rgb1': '0.3 0.35 0.4',
        'background_rgb2': '0.2 0.25 0.3'
    },
    {
        'background_builtin': 'flat',
        'background_rgb1': '0.4 0.3 0.35',
        'background_rgb2': '0.3 0.2 0.25'
    },
    {
        'background_builtin': 'flat',
        'background_rgb1': '0.25 0.3 0.35',
        'background_rgb2': '0.15 0.2 0.25'
    },
    {
        'background_builtin': 'flat',
        'background_rgb1': '0.35 0.4 0.45',
        'background_rgb2': '0.25 0.3 0.35'
    },
    {
        'background_builtin': 'flat',
        'background_rgb1': '0.45 0.35 0.4',
        'background_rgb2': '0.35 0.25 0.3'
    },
    {
        'background_builtin': 'flat',
        'background_rgb1': '0.4 0.4 0.45',
        'background_rgb2': '0.3 0.3 0.35'
    },
    {
        'background_builtin': 'flat',
        'background_rgb1': '0.35 0.45 0.4',
        'background_rgb2': '0.25 0.35 0.3'
    },
    {
        'background_builtin': 'flat',
        'background_rgb1': '0.5 0.4 0.35',
        'background_rgb2': '0.4 0.3 0.25'
    },
    {
        'background_builtin': 'flat',
        'background_rgb1': '0.3 0.4 0.5',
        'background_rgb2': '0.2 0.3 0.4'
    },
    {
        'background_builtin': 'flat',
        'background_rgb1': '0.45 0.3 0.4',
        'background_rgb2': '0.35 0.2 0.3'
    },
    {
        'background_builtin': 'flat',
        'background_rgb1': '0.4 0.5 0.45',
        'background_rgb2': '0.3 0.4 0.35'
    },
    {
        'background_builtin': 'flat',
        'background_rgb1': '0.35 0.3 0.4',
        'background_rgb2': '0.25 0.2 0.3'
    },
]


# =================
# DISTRACTOR CONFIG
# =================

# distractor regions
# Maze coordinates: x ~ [0.4, 3.2], y ~ [0.6, 3.2]
# We place distractors well outside these bounds
_MAZE_MAX = 5.5
_MAZE_MIN_X = 0.4
_MAZE_MIN_Y = 0.6
_PAD = 0.6
DISTRACTOR_REGIONS = [
    # Top-right: x > maze_max, y > maze_max
    ("top_right", (_MAZE_MAX + 0.1, _MAZE_MAX + _PAD), (_MAZE_MAX + 0.1, _MAZE_MAX + _PAD)),
    # Top-left: x < maze_min, y > maze_max
    ("top_left", (-_PAD - 1.0, _MAZE_MIN_X - 0.1), (_MAZE_MAX + 0.1, _MAZE_MAX + _PAD)),
    # Bottom-right: x > maze_max, y < maze_min
    ("bottom_right", (_MAZE_MAX + 0.1, _MAZE_MAX + _PAD), (-_PAD - 1.0, _MAZE_MIN_Y - 0.1)),
    # Bottom-left: x < maze_min, y < maze_min
    ("bottom_left", (-_PAD - 1.0, _MAZE_MIN_X - 0.1), (-_PAD - 1.0, _MAZE_MIN_Y - 0.1)),
]
DISTRACTOR_Z = 0.02


def sample_distractor_xy(rng: np.random.Generator) -> Tuple[float, float]:
    """
    Pick one of four outside regions (TL/TR/BL/BR) and sample (x,y) in it.
    
    Args:
        rng: NumPy random generator
        
    Returns:
        Tuple of (x, y) coordinates for distractor position
    """
    idx = int(rng.integers(0, 4))
    _, (x_lo, x_hi), (y_lo, y_hi) = DISTRACTOR_REGIONS[idx]
    x = float(rng.uniform(x_lo, x_hi))
    y = float(rng.uniform(y_lo, y_hi))
    return x, y


def sample_distractor_rgba(rng: np.random.Generator) -> Tuple[float, float, float, float]:
    """
    Sample a random color for distractor, avoiding black (in test set) and predominantly red/green (start/goal color)
    
    Args:
        rng: NumPy random generator
        
    Returns:
        Tuple of (r, g, b, a) color values
    """
    for _ in range(50):
        rgb = rng.uniform(0.35, 0.95, size=3)
        r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
        predominantly_red = r > g and r > b and r > 0.5
        predominantly_green = g > r and g > b and g > 0.5
        is_too_dark = r < 0.2 and g < 0.2 and b < 0.2
        if not predominantly_red and not predominantly_green and not is_too_dark:
            return r, g, b, 1.0
    # fallback to safe color
    return 0.85, 0.2, 0.75, 1.0


def sample_dr_background(rng: np.random.Generator) -> Dict[str, str]:
    """
    Sample a random background configuration from the DR set.
    
    Args:
        rng: NumPy random generator
        
    Returns:
        Dictionary with background_builtin, background_rgb1, background_rgb2
    """
    idx = int(rng.integers(0, len(DR_BACKGROUNDS)))
    return DR_BACKGROUNDS[idx].copy()


def is_test_background(bg_config: Dict[str, str]) -> bool:
    """
    Check if a background config matches any of the test set backgrounds.
    
    Args:
        bg_config: Background configuration dictionary
        
    Returns:
        True if the config matches a test background
    """
    for test_bg in TEST_BACKGROUNDS.values():
        if (bg_config['background_builtin'] == test_bg['background_builtin'] and
            bg_config['background_rgb1'] == test_bg['background_rgb1'] and
            bg_config['background_rgb2'] == test_bg['background_rgb2']):
            return True
    return False


def is_test_distractor(rgba: Tuple[float, float, float, float], threshold: float = 0.1) -> bool:
    """
    Check if a distractor color is too close to the test distractor (black).
    
    Args:
        rgba: Color tuple (r, g, b, a)
        threshold: Maximum value for each RGB component to be considered "black"
        
    Returns:
        True if the color is too close to black
    """
    r, g, b, _ = rgba
    return r < threshold and g < threshold and b < threshold
