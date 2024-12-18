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

# def get_scan(slices, label):
#     l = []
#     for row in slices:
#         l.append(row[label])
    
#     return l

# ssdd = np.arange(1, 21).reshape((4, 5))
# ssdd_tensor = np.zeros((4, 5, 3), dtype=ssdd.dtype)
# ssdd_tensor[:, :, 0] = ssdd
# ssdd_tensor[:, :, 1] = ssdd + 20
# ssdd_tensor[:, :, 2] = ssdd + 40

# direction = 8
# slices = solution.scan_slices(ssdd_tensor, direction)
# print(ssdd_tensor[:, :, 1])
# print(get_scan(slices, 1))

# rows_num = 4
# cols_num = 5
# flat_indices = np.arange(0, rows_num * cols_num).reshape((rows_num, cols_num))
# l = np.zeros_like(flat_indices)
# print(flat_indices)
# flat_indices_expanded = np.expand_dims(flat_indices, axis=-1)
# flat_indices = solution.scan_slices(flat_indices_expanded, 8)
# for slice_idx in range(len(flat_indices)):
#     scan_line_flat_indices = flat_indices[slice_idx]
#     squeezed = np.squeeze(scan_line_flat_indices, axis=0)  # Specify axis=1 to remove the size-1 middle dimension
#     indices = np.unravel_index(squeezed, (rows_num, cols_num))

#     print(indices)

#     l[indices] = slice_idx

# print(l)


from PIL import Image

def load_resize_save_image(input_path: str, output_path: str, resize_factor: float) -> None:
    """
    Load an image, resize it by a given factor, and save it as a PNG.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path to save the resized image as PNG.
        resize_factor (float): Factor by which to resize the image (e.g., 0.5 for half size, 2 for double size).

    Returns:
        None
    """
    if resize_factor <= 0:
        raise ValueError("Resize factor must be greater than 0.")

    # Load the image
    image = Image.open(input_path)

    # Calculate the new size
    new_size = (int(image.width * resize_factor), int(image.height * resize_factor))

    # Resize the image
    resized_image = image.resize(new_size, Image.ANTIALIAS)

    # Save the resized image as PNG
    resized_image.save(output_path, format="PNG")

# Example usage:
load_resize_save_image("my_image_left_original.png", "my_image_left.png", 0.25)
load_resize_save_image("my_image_right_original.png", "my_image_right.png", 0.25)