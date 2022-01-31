import numpy as np
import torch

# pts (numpy array): [batch_size, 3] with 0 <= x, y, z <= 1 (i.e. in a unit box grid) to be interpolated (otherwise, if outside the unit box grid, extrapolated)
# features (torch tensor): [D, H, W, feature_len] with D, H, W >= 2
# output: [batch_size, feature_len]
def interpolate_trilinear(pts, features):
    depth, height, width, _ = features.shape

    # (i, j, k) are the indices of the (lower, bottom, down) feature, out of the 8 features we interpolate point between. (0, 0, 0) <= (i, j, k) <= (D - 2, H - 2, W - 2)
    pts = np.copy(pts)
    pts[:, 0] *= depth - 1
    pts[:, 1] *= height - 1
    pts[:, 2] *= width - 1
    ijk = pts.astype(np.int32)
    i = np.clip(ijk[:, 0], 0, depth - 2, out=ijk[:, 0])
    j = np.clip(ijk[:, 1], 0, height - 2, out=ijk[:, 1])
    k = np.clip(ijk[:, 2], 0, width - 2, out=ijk[:, 2])

    # features at each of the 8 corners
    f000 = features[i   , j   ,  k   ]
    f100 = features[(i+1), j   ,  k   ]
    f010 = features[ i   ,(j+1),  k   ]
    f001 = features[ i   , j   , (k+1)]
    f101 = features[(i+1), j   , (k+1)]
    f011 = features[ i   ,(j+1), (k+1)]
    f110 = features[(i+1),(j+1),  k   ]
    f111 = features[(i+1),(j+1), (k+1)]

    # (x, y, z) are the "sliders" for the three dimensions, with (0, 0, 0) being full weights for the (lower, bottom, down) feature and (1, 1, 1) being full weights for the (upper, top, up) feature
    xyz = torch.Tensor(pts - ijk)
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    fxyz = (f000.T * (1 - x)*(1 - y)*(1 - z)
            + f100.T * x * (1 - y) * (1 - z)
            + f010.T * (1 - x) * y * (1 - z)
            + f001.T * (1 - x) * (1 - y) * z
            + f101.T * x * (1 - y) * z
            + f011.T * (1 - x) * y * z
            + f110.T * x * y * (1 - z)
            + f111.T * x * y * z).T
    return fxyz

feats = torch.zeros((5, 5, 5, 8))
feats[3, 2, 0] = 1
feats[4, 4, 4] = 1
xyz = np.array([[0.8, 0.71, 0.19], [1, 1, 1], [1, 0.8, 0.95], [1.1, 1.01, 1.05]])
print(interpolate_trilinear(xyz, feats))