# Code to check whether simplicial complex has a hole
#
# What other tasks require 3D scene understanding?

import numpy as np


def has_bottom_hole_3x3x3(voxels):
    # For a 3x3x3 block of voxels, returns True if the cubical complex does not
    # have a hole passing through the bottom (z = 0) face, otherwise False
    
    # 4 cells on edges of bottom face must be filled
    filled = [(1, 0, 0), (0, 1, 0), (1, 2, 0), (2, 1, 0)]
    if not all(voxels[coord] for coord in filled):
        return False

    # bottom center and middle center must be empty
    if voxels[1, 1, 0] or voxels[1, 1, 1]:
        return False

    # need just one of the other faces to be open
    faces = [(1, 0, 1), (0, 1, 1), (1, 2, 1), (2, 1, 1), (1, 1, 2)]
    if any(not voxels[coord] for coord in faces):
        return True

    # have a depression, but no hole
    return False

def has_hole_3x3x3(voxels):
    # For a 3x3x3 of voxels, returns True if the cubical complex is of non-zero
    # genus. This can be done more generally for cubical complexes of any size
    # using cubical homology (maybe implement later).

    # Algorithm: alternately rotate voxels so each of the 6 faces is at the
    # bottom and check for hole (brute force)
    transforms = [
        (0, (0, 1)),
        (1, (1, 2)),
        (2, (1, 2)),
        (3, (1, 2)),
        (1, (0, 2)),
        (3, (0, 2)),
    ]
    for k, axes in transforms:
        rotated_voxels = np.rot90(voxels, k=k, axes=axes)
        has_hole = has_bottom_hole_3x3x3(rotated_voxels)
        if has_hole:
            return True
    return False
