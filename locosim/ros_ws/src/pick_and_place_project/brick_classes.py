#!/usr/bin/env python3
"""
Brick class definitions for PAPA project
Shared between object spawner and perception modules
"""

import numpy as np


# Available brick types with their STL mesh files and properties
BRICK_CLASSES = {
    'X1-Y1-Z2': {
        'mesh': 'X1-Y1-Z2.stl',
        'size': np.array([0.008, 0.008, 0.016]),  # Approximate dimensions in meters
        'mass': 0.05,
        'class': 'small_cube',
    },
    'X1-Y2-Z1': {
        'mesh': 'X1-Y2-Z1.stl',
        'size': np.array([0.008, 0.016, 0.008]),
        'mass': 0.05,
        'class': 'flat_rectangle',
    },
    'X1-Y2-Z2': {
        'mesh': 'X1-Y2-Z2.stl',
        'size': np.array([0.008, 0.016, 0.016]),
        'mass': 0.08,
        'class': 'rectangle',
    },
    'X1-Y3-Z2': {
        'mesh': 'X1-Y3-Z2.stl',
        'size': np.array([0.008, 0.024, 0.016]),
        'mass': 0.10,
        'class': 'long_rectangle',
    },
    'X1-Y4-Z2': {
        'mesh': 'X1-Y4-Z2.stl',
        'size': np.array([0.008, 0.032, 0.016]),
        'mass': 0.12,
        'class': 'very_long_rectangle',
    },
    'X2-Y2-Z2': {
        'mesh': 'X2-Y2-Z2.stl',
        'size': np.array([0.016, 0.016, 0.016]),
        'mass': 0.15,
        'class': 'large_cube',
    },
    'X1-Y2-Z2-CHAMFER': {
        'mesh': 'X1-Y2-Z2-CHAMFER.stl',
        'size': np.array([0.008, 0.016, 0.016]),
        'mass': 0.08,
        'class': 'chamfered_rectangle',
    },
    'X1-Y2-Z2-TWINFILLET': {
        'mesh': 'X1-Y2-Z2-TWINFILLET.stl',
        'size': np.array([0.008, 0.016, 0.016]),
        'mass': 0.08,
        'class': 'filleted_rectangle',
    },
}
