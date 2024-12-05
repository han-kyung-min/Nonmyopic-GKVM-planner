
import cv2
import numpy as np

UNKNOWN = 128
FREE = 0
OCCUPIED = 255

def load_gkvm_map_from_gmapimg( filename, ukn = 205, free = 254, occ = 0, num_ds = 0 ):
    gmap_img = cv2.imread(filename, 0)
    #gmap_img = gmap_img[25:-25, 25:-25]
    # enforce map to 0 (free), 128 (unk), 255 (occ)
    unk_idx = np.where(gmap_img == ukn)
    free_idx = np.where(gmap_img == free)
    occ_idx = np.where(gmap_img == occ)
    gmap = np.ones( gmap_img.shape ) * UNKNOWN
    gmap[free_idx] = FREE
    gmap[occ_idx] = OCCUPIED

    if num_ds > 0:
        for ii in range(0, num_ds):
            gmap = cv2.pyrDown(gmap)
            # Enforce img to be bounded
            unk_idx = np.where( (gmap > (UNKNOWN - 48)) & (gmap < (UNKNOWN + 48)) )
            free_idx = np.where( gmap < (FREE + 48) )
            occ_idx = np.where(  gmap >  (OCCUPIED - 48) )
            gmap[free_idx] = FREE
            gmap[occ_idx] = OCCUPIED
            gmap[unk_idx] = UNKNOWN

    (rows, cols) = gmap.shape
    observation = np.zeros([rows, cols] )
    uncertainty = np.ones([rows, cols] ) * 0.1
    free_idx = np.argwhere( gmap < FREE + 48 )
    occ_idx = np.argwhere( gmap > OCCUPIED - 48 )

    for ii in range(0, len(free_idx) ):
        y, x = free_idx[ii]
        observation[y][x] = -1
        uncertainty[y][x] = 2.0

    # for ii in range(0, len(occ_idx) ):
    #     y, x = occ_idx[ii]
    #     observation[y][x] = 0

    return observation, uncertainty