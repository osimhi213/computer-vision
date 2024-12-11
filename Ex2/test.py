import numpy as np
import matplotlib.image as mpimg

from solution import Solution


def load_data(is_your_data=False):
    # Read the data:
    if is_your_data:
        left_image = mpimg.imread('my_image_left.png')
        right_image = mpimg.imread('my_image_right.png')
    else:
        left_image = mpimg.imread('image_left.png')
        right_image = mpimg.imread('image_right.png')
    return left_image, right_image


solution = Solution()

# left_image, right_image = load_data()

# ssdd_tensor = solution.ssd_distance(left_image.astype(np.float64),
#                                 right_image.astype(np.float64),
#                                 win_size=3,
#                                 dsp_range=20)

# dp_labeling = solution.dp_labeling(ssdd_tensor, 0.5, 3)
# print(dp_labeling)

# t = ssdd_tensor[:5, :4, 21].astype(np.int32)
# t = np.arange(0, 20).reshape((4, 5))
# print(np.diagonal(t))

ssdd = np.arange(1, 21).reshape((5, 4))
ssdd_tensor = np.zeros((5, 4, 3), dtype=ssdd.dtype)
ssdd_tensor[:, :, 0] = ssdd
ssdd_tensor[:, :, 1] = ssdd + 20
ssdd_tensor[:, :, 2] = ssdd + 40

direction = 1
slices = solution.scan_slices(ssdd_tensor, direction)
print(direction)
print(ssdd_tensor[:, :, 0])
print(slices[:, :, 0])
print('-------------------------------')
print(ssdd_tensor[:, :, 1])
print(slices[:, :, 1])
print('-------------------------------')
print(ssdd_tensor[:, :, 2])
print(slices[:, :, 2])