#!/usr/bin/env python3
"""
Brick class definitions for PAPA project
Shared between object spawner and perception modules
"""

import numpy as np

# Available brick types with their STL mesh files and properties
# Dimensions based on STL Bounding Boxes (including studs)
# The 'id' attribute is used for YOLO training
BRICK_CLASSES = {
    'X1-Y1-Z2': {
        'id': 0,
        'class': 'small_cube',
        'mesh': 'X1-Y1-Z2.stl',
        'size': np.array([0.03, 0.03, 0.06]),
        'mass': 0.05,
    },
    'X1-Y2-Z1': {
        'id': 1,
        'class': 'flat_rectangle',
        'mesh': 'X1-Y2-Z1.stl',
        'size': np.array([0.03, 0.06, 0.04]),
        'mass': 0.05,
    },
    'X1-Y2-Z2': {
        'id': 2,
        'class': 'rectangle',
        'mesh': 'X1-Y2-Z2.stl',
        'size': np.array([0.03, 0.06, 0.06]),
        'mass': 0.08,
    },
    'X1-Y2-Z2-CHAMFER': {
        'id': 3,
        'class': 'rectangle_chamfered',
        'mesh': 'X1-Y2-Z2-CHAMFER.stl',
        'size': np.array([0.03, 0.06, 0.06]),
        'mass': 0.08,
    },
    'X1-Y2-Z2-TWINFILLET': {
        'id': 4,
        'class': 'rectangle_filleted',
        'mesh': 'X1-Y2-Z2-TWINFILLET.stl',
        'size': np.array([0.03, 0.06, 0.06]),
        'mass': 0.08,
    },
    'X1-Y3-Z2': {
        'id': 5,
        'class': 'long_rectangle',
        'mesh': 'X1-Y3-Z2.stl',
        'size': np.array([0.03, 0.10, 0.06]),
        'mass': 0.10,
    },
    'X1-Y3-Z2-FILLET': {
        'id': 6,
        'class': 'long_rectangle_filleted',
        'mesh': 'X1-Y3-Z2-FILLET.stl',
        'size': np.array([0.03, 0.10, 0.06]),
        'mass': 0.10,
    },
    'X1-Y4-Z1': {
        'id': 7,
        'class': 'flat_very_long_rectangle',
        'mesh': 'X1-Y4-Z1.stl',
        'size': np.array([0.03, 0.13, 0.04]),
        'mass': 0.06,
    },
    'X1-Y4-Z2': {
        'id': 8,
        'class': 'very_long_rectangle',
        'mesh': 'X1-Y4-Z2.stl',
        'size': np.array([0.03, 0.13, 0.06]),
        'mass': 0.12,
    },
    'X2-Y2-Z2': {
        'id': 9,
        'class': 'large_cube',
        'mesh': 'X2-Y2-Z2.stl',
        'size': np.array([0.06, 0.06, 0.06]),
        'mass': 0.15,
    },
    'X2-Y2-Z2-FILLET': {
        'id': 10,
        'class': 'large_cube_filleted',
        'mesh': 'X2-Y2-Z2-FILLET.stl',
        'size': np.array([0.06, 0.06, 0.06]),
        'mass': 0.15,
    },
}
