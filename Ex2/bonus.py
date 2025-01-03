import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from main import load_data
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import sobel

from solution import Solution
from scipy.ndimage import uniform_filter

def compute_ssim(left_image: np.ndarray, right_image: np.ndarray, win_size: int) -> np.ndarray:
    """
    Compute SSIM (Structural Similarity Index) map for two images using a sliding window approach.

    Args:
        left_image (np.ndarray): The first image (can be RGB or grayscale).
        right_image (np.ndarray): The second image (can be RGB or grayscale).
        win_size (int): Size of the sliding window (must be an odd number).

    Returns:
        np.ndarray: SSIM map with the same shape as the input images.
    """
    assert left_image.shape == right_image.shape, "Input images must have the same shape"
    assert win_size % 2 == 1, f"Window size must be odd, got {win_size}"

    # Convert to grayscale if input is RGB
    if left_image.ndim == 3 and left_image.shape[2] == 3:  # RGB image
        left_image = 0.2989 * left_image[:, :, 0] + 0.5870 * left_image[:, :, 1] + 0.1140 * left_image[:, :, 2]
        right_image = 0.2989 * right_image[:, :, 0] + 0.5870 * right_image[:, :, 1] + 0.1140 * right_image[:, :, 2]

    C1 = 0.01 ** 2  # Constant to stabilize division (typically small, derived from L=1 in [0,1] range)
    C2 = 0.03 ** 2  # Constant to stabilize division (typically small, derived from L=1 in [0,1] range)

    # Ensure the kernel size matches the input rank
    kernel = (win_size,) * left_image.ndim

    # Means
    mu1 = uniform_filter(left_image, size=kernel)
    mu2 = uniform_filter(right_image, size=kernel)

    # Squares of means
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Variance and covariance
    sigma1_sq = uniform_filter(left_image ** 2, size=kernel) - mu1_sq
    sigma2_sq = uniform_filter(right_image ** 2, size=kernel) - mu2_sq
    sigma12 = uniform_filter(left_image * right_image, size=kernel) - mu1_mu2

    # SSIM map computation
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + 1e-12)  # Add small epsilon to avoid division by zero

    return ssim_map

def compute_gmd(left_image: np.ndarray, right_image: np.ndarray, win_size: int = 3) -> np.ndarray:
    """
    Compute Gradient Magnitude Difference (GMD) for two images, considering window size and handling multi-channel images.

    Args:
        left_image (np.ndarray): The first image.
        right_image (np.ndarray): The second image.
        win_size (int): The size of the window for local smoothing.

    Returns:
        np.ndarray: GMD map with the same shape as the input images.
    """
    # Initialize an empty list to store GMD results for each channel
    gmd_channels = []

    # Loop through each channel (assuming multi-channel images like RGB)
    for i in range(left_image.shape[-1]):
        # Extract the i-th channel from both images
        left_channel = left_image[..., i]
        right_channel = right_image[..., i]

        # Compute gradient magnitudes for the current channel
        left_grad_x = sobel(left_channel, axis=1)
        left_grad_y = sobel(left_channel, axis=0)
        left_grad_mag = np.sqrt(left_grad_x**2 + left_grad_y**2)

        right_grad_x = sobel(right_channel, axis=1)
        right_grad_y = sobel(right_channel, axis=0)
        right_grad_mag = np.sqrt(right_grad_x**2 + right_grad_y**2)

        # Compute Gradient Magnitude Difference for the current channel
        gmd_map_channel = np.abs(left_grad_mag - right_grad_mag)

        # Optionally smooth the GMD map for local differences (using a uniform filter as an example)
        gmd_map_channel = uniform_filter(gmd_map_channel, size=win_size)

        # Append the GMD map for this channel
        gmd_channels.append(gmd_map_channel)

    # Sum over all channels to get the final GMD map (or use another aggregation strategy)
    gmd_map = np.sum(gmd_channels, axis=0)

    return gmd_map

def ssim_distance(left_image: np.ndarray,
                            right_image: np.ndarray,
                            win_size: int,
                            dsp_range: int) -> np.ndarray:
    """Compute SSIM distances tensor with optimized convolution."""
    num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
    disparity_values = range(-dsp_range, dsp_range+1)
    distance_tensor = np.zeros((num_of_rows,
                            num_of_cols,
                            len(disparity_values)))

    assert win_size % 2 == 1, f'win_size has to be an odd number, got {win_size}'

    pad_width = [(0, 0), (dsp_range, dsp_range)] + [(0, 0)] * (len(right_image.shape) - 2)
    right_image_padded = np.pad(right_image, pad_width)

    for i, disp in enumerate(disparity_values):
        x_d = dsp_range + disp
        right_image_shifted = right_image_padded[:, x_d:x_d + num_of_cols]

        distance = compute_gmd(left_image, right_image_shifted)
        distance_tensor[:, :, i] = distance

    distance_tensor -= distance_tensor.min()
    distance_tensor /= distance_tensor.max()
    distance_tensor *= 255.0

    return distance_tensor

def main():
    COST1 = 13.5
    COST2 = 39.0
    WIN_SIZE = 3
    DISPARITY_RANGE = 20
    solution = Solution()
    left_image, right_image = load_data()
    ssim = ssim_distance(left_image.astype(np.float64),
                         right_image.astype(np.float64),
                         win_size=WIN_SIZE,
                         dsp_range=DISPARITY_RANGE)
    label_map = solution.sgm_labeling(ssim, COST1, COST2)

    # plot the left image and the estimated depth
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 2, 2)
    plt.imshow(label_map)
    plt.colorbar()
    plt.title('Naive Depth')
    plt.savefig('bonus_outputs/naive_labeling.jpg')


if __name__ == "__main__":
    main()

