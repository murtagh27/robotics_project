#!/usr/bin/env python3
"""
@file brick_classes.py
@brief Brick class definitions for PAPA project.
@details Shared between object spawner and perception modules. Defines all available brick types
         with their physical properties, STL mesh files, dimensions, and YOLO class IDs.
@author Benjamin Krech
@date January 2026
"""

import numpy as np

"""
@var BRICK_CLASSES
@brief Dictionary containing all available brick types with their properties.
@details Each brick type is identified by its dimensional code (e.g., 'X1-Y2-Z2') and contains:
         - id: Unique integer ID used for YOLO training and detection (0-10).
         - class: Trivial name for identifying the bricks, based on their appearance.
         - mesh: Filename of the STL mesh in the brick_description package.
         - size: 3D dimensions [x, y, z] in meters as numpy array (including studs).
         - mass: Mass in kilograms for physics simulation.
         
         - The dimensional code format is X[width]-Y[length]-Z[height] where 1 unit ≈ 0.03m.
         - Possible suffixes: CHAMFER, FILLET, TWINFILLET indicate edge modifications.
"""
BRICK_CLASSES = {
    'X1-Y1-Z2': {
        'id': 0,
        'class': 'small_cube',
        'mesh': 'X1-Y1-Z2.stl',
        'size': np.array([0.03, 0.03, 0.06]),
        'mass': 0.20,
    },
    'X1-Y2-Z1': {
        'id': 1,
        'class': 'flat_rectangle',
        'mesh': 'X1-Y2-Z1.stl',
        'size': np.array([0.03, 0.06, 0.04]),
        'mass': 0.20,
    },
    'X1-Y2-Z2': {
        'id': 2,
        'class': 'rectangle',
        'mesh': 'X1-Y2-Z2.stl',
        'size': np.array([0.03, 0.06, 0.06]),
        'mass': 0.30,
    },
    'X1-Y2-Z2-CHAMFER': {
        'id': 3,
        'class': 'rectangle_chamfered',
        'mesh': 'X1-Y2-Z2-CHAMFER.stl',
        'size': np.array([0.03, 0.06, 0.06]),
        'mass': 0.30,
    },
    'X1-Y2-Z2-TWINFILLET': {
        'id': 4,
        'class': 'rectangle_filleted',
        'mesh': 'X1-Y2-Z2-TWINFILLET.stl',
        'size': np.array([0.03, 0.06, 0.06]),
        'mass': 0.30,
    },
    'X1-Y3-Z2': {
        'id': 5,
        'class': 'long_rectangle',
        'mesh': 'X1-Y3-Z2.stl',
        'size': np.array([0.03, 0.10, 0.06]),
        'mass': 0.40,
    },
    'X1-Y3-Z2-FILLET': {
        'id': 6,
        'class': 'long_rectangle_filleted',
        'mesh': 'X1-Y3-Z2-FILLET.stl',
        'size': np.array([0.03, 0.10, 0.06]),
        'mass': 0.40,
    },
    'X1-Y4-Z1': {
        'id': 7,
        'class': 'flat_very_long_rectangle',
        'mesh': 'X1-Y4-Z1.stl',
        'size': np.array([0.03, 0.13, 0.04]),
        'mass': 0.25,
    },
    'X1-Y4-Z2': {
        'id': 8,
        'class': 'very_long_rectangle',
        'mesh': 'X1-Y4-Z2.stl',
        'size': np.array([0.03, 0.13, 0.06]),
        'mass': 0.50,
    },
    'X2-Y2-Z2': {
        'id': 9,
        'class': 'large_cube',
        'mesh': 'X2-Y2-Z2.stl',
        'size': np.array([0.06, 0.06, 0.06]),
        'mass': 0.60,
    },
    'X2-Y2-Z2-FILLET': {
        'id': 10,
        'class': 'large_cube_filleted',
        'mesh': 'X2-Y2-Z2-FILLET.stl',
        'size': np.array([0.06, 0.06, 0.06]),
        'mass': 0.60,
    },
}
