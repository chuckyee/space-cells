import numpy as np

from geometry import has_bottom_hole_3x3x3, has_hole_3x3x3

def test_has_bottom_hole_3x3x3():
    # coordinates are value[x, y, z]
    z0 = [[1, 1, 1],
          [1, 0, 1],
          [0, 1, 1]]
    z1 = [[1, 0, 1],
          [1, 0, 1],
          [1, 1, 1]]
    z2 = [[1, 1, 1],
          [1, 1, 1],
          [1, 1, 1]]
    voxels = np.zeros((3, 3, 3), dtype=bool)
    voxels[:,:,0] = z0
    voxels[:,:,1] = z1
    voxels[:,:,2] = z2

    assert has_bottom_hole_3x3x3(voxels)

def test_has_hole_3x3x3():
    z0 = [[0, 1, 0],
          [1, 0, 0],
          [0, 0, 0]]
    z1 = [[0, 0, 0],
          [0, 0, 1],
          [0, 1, 0]]
    z2 = [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]]
    voxels = np.zeros((3, 3, 3), dtype=bool)
    voxels[:,:,0] = z0
    voxels[:,:,1] = z1
    voxels[:,:,2] = z2

    assert has_hole_3x3x3(voxels)
