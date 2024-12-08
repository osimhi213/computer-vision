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

left_image, right_image = load_data()

ssdd = solution.ssd_distance(left_image.astype(np.float64),
                                right_image.astype(np.float64),
                                win_size=3,
                                dsp_range=20)
solution.dp_grade_slice(ssdd[0].T, 0.5, 3)

