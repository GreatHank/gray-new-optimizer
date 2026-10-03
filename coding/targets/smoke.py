"""Create a small three-channel target that includes diffraction order (0, 0)."""

import numpy as np
import scipy.io as sio

from coding import ROOT

def main():
    output = ROOT/"output/targets/zero_order_smoke/target.mat"
    output.parent.mkdir(parents=True, exist_ok=True)
    targets = np.zeros((3, 32, 32), dtype=np.float32)
    targets[0, 8:24, 8:24] = 1.0
    targets[1, 9:23, 6:13] = 2 / 3
    targets[2, 6:13, 9:23] = 1 / 3
    positions = np.asarray([[0, 0], [0, 1], [1, 0]], dtype=np.int16)
    sio.savemat(output, {"bw_all": targets, "grid_positions": positions, "order_pairs": positions, "gray_level_count": 4}, do_compression=True)
    print(output)


if __name__ == "__main__":
    main()
