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

def get_scan(slices, label):
    l = []
    for row in slices:
        l.append(row[label])
    
    return l

ssdd = np.arange(1, 21).reshape((4, 5))
# ssdd_tensor = np.zeros((4, 5, 3), dtype=ssdd.dtype)
# ssdd_tensor[:, :, 0] = ssdd
# ssdd_tensor[:, :, 1] = ssdd + 20
# ssdd_tensor[:, :, 2] = ssdd + 40

# direction = 8
# slices = solution.scan_slices(ssdd_tensor, direction)
# print(ssdd_tensor[:, :, 1])
# print(get_scan(slices, 1))

rows_num = 4
cols_num = 5
flat_indices = np.arange(0, rows_num * cols_num).reshape((rows_num, cols_num))
l = np.zeros_like(flat_indices)
print(flat_indices)
flat_indices_expanded = np.expand_dims(flat_indices, axis=-1)
flat_indices = solution.scan_slices(flat_indices_expanded, 8)
for slice_idx in range(len(flat_indices)):
    scan_line_flat_indices = flat_indices[slice_idx]
    squeezed = np.squeeze(scan_line_flat_indices, axis=0)  # Specify axis=1 to remove the size-1 middle dimension
    indices = np.unravel_index(squeezed, (rows_num, cols_num))

    print(indices)

    l[indices] = slice_idx

print(l)
