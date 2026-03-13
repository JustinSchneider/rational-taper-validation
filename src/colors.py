"""Centralized color constants for tapered-model-comparison plots."""

# --- Model colors ---
MODEL_COLORS = {
    'rational_taper': 'steelblue',
    'nfw':            'darkorange',
    'mond_fixed':     'mediumpurple',
    'mond_free':      'indigo',
}

# Human-readable display labels
MODEL_LABELS = {
    'rational_taper': 'Rational Taper',
    'nfw':            'NFW',
    'mond_fixed':     'MOND (fixed)',
    'mond_free':      'MOND (free)',
}

# Line styles used in rotation curve plots
MODEL_LINESTYLES = {
    'rational_taper': '-',
    'nfw':            '--',
    'mond_fixed':     ':',
    'mond_free':      '-.',
}

# --- Quality flag colors ---
QFLAG_COLORS = {1: 'teal', 2: 'goldenrod', 3: 'red'}

# --- Observational component colors ---
OBS_COLORS = {
    'v_obs':   'black',
    'v_bary':  'dimgray',
    'v_gas':   'green',
    'v_disk':  'red',
    'v_bulge': 'magenta',
}

# --- Continuous colormaps ---
CMAP_MASS = 'magma'   # for luminosity/mass color axes
