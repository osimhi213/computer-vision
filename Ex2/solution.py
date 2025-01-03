"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        """INSERT YOUR CODE HERE"""
        assert win_size % 2 == 1, f'win_size has to be an odd number, got {win_size}'

        pad_width = [(0, 0), (dsp_range, dsp_range)] + [(0, 0)] * (len(right_image.shape) - 2)
        right_image_padded = np.pad(right_image, pad_width)

        for i, disp in enumerate(disparity_values):
            x_d = dsp_range + disp
            images_diff = left_image - right_image_padded[:, x_d:x_d+num_of_cols]
            images_square_diff = np.sum(images_diff ** 2, axis=2)
            sum_operator = np.ones((win_size, win_size))
            ssd = convolve2d(images_square_diff, sum_operator, mode='same')
            ssdd_tensor[:, :, i] = ssd
 
        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0

        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        """INSERT YOUR CODE HERE"""
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)

        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))
        """INSERT YOUR CODE HERE"""
        # first column initialization     
        l_slice[:, 0] = c_slice[:, 0]

        for col in range(1, num_of_cols):
            # Assume p1 and p2 are positive, with p1 <= p2, such that taking the minimum of all second neighbors plus p2
            # is equivalent to taking the minimum of all disparity values.
            prev_col = l_slice[:, col-1]
            left_neighbor = p1 + np.append(np.inf, prev_col[:-1])
            right_neighbor = p1 + np.append(prev_col[1:], np.inf)
            second_neighbors = p2 + np.repeat(np.min(prev_col), num_labels)
            min_costs = np.min(np.array([prev_col, left_neighbor, right_neighbor, second_neighbors]), axis=0)
            
            l_slice[:, col] = c_slice[:, col] + min_costs - np.min(prev_col)

        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        for row in range(ssdd_tensor.shape[0]):
            l[row] = self.dp_grade_slice(ssdd_tensor[row].T, p1, p2).T

        return self.naive_labeling(l)

    def scan_diagonals(self, tensor, direction) -> np.ndarray:
        num_of_rows, num_of_cols = tensor.shape[:2]
        offset_min = -num_of_rows + 1
        offset_max = num_of_cols - 1
        slices = []

        for offset in range(offset_min, offset_max + 1):
            step = -1 if direction > 4 else 1
            # leveraging symmetry
            diagonal = tensor.diagonal(offset)[:, ::step]
            slices.append(diagonal)

        return slices 

    def scan_slices(self, tensor: np.ndarray, direction: int) -> np.ndarray:
        """Return the tensor slices scanned along the given direction.
        """
        assert 1 <= direction <= 8, 'scan direction should be an integer between 1 and 8'
        scan_line = direction % 4
        tensor_to_scan = tensor

        if scan_line == 1 or scan_line == 3:  # lines
            if scan_line == 3:  # direction 3/7
                tensor_to_scan = tensor.transpose(1, 0, 2)

            c_slices = tensor_to_scan.transpose(0, 2, 1)
            if direction > 4:  # leveraging symmetry 
                c_slices = list(c_slices[:, :, ::-1])
        else:  # diagonal
            if scan_line == 0:  # direction 4/8
                tensor_to_scan = tensor.transpose(1, 0, 2)[::-1, :, :]
      
            c_slices = self.scan_diagonals(tensor_to_scan, direction)

        return c_slices

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}            
        """INSERT YOUR CODE HERE"""
        num_of_rows, num_of_cols = ssdd_tensor.shape[:2]
        flat_indices = np.arange(0, num_of_rows * num_of_cols).reshape((num_of_rows, num_of_cols, 1))

        for direction in range(1, num_of_directions + 1):
            l = np.zeros_like(ssdd_tensor)
            direction_slices = self.scan_slices(ssdd_tensor, direction)
            direction_flat_indices = self.scan_slices(flat_indices, direction)

            for slice_idx, c_slice in enumerate(direction_slices):
                slice_indices = np.unravel_index(
                    np.squeeze(direction_flat_indices[slice_idx], axis=0),
                    (num_of_rows, num_of_cols)
                )

                l_slice = self.dp_grade_slice(c_slice, p1, p2).T 
                l[slice_indices] = l_slice
            
            direction_to_slice[direction] = self.naive_labeling(l)

        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        l_direction_tensors = np.zeros((num_of_directions, *ssdd_tensor.shape))
        num_of_rows, num_of_cols = ssdd_tensor.shape[:2]
        flat_indices = np.arange(0, num_of_rows * num_of_cols).reshape((num_of_rows, num_of_cols, 1))

        for i, direction in enumerate(range(1, num_of_directions + 1)):
            direction_slices = self.scan_slices(ssdd_tensor, direction)
            direction_flat_indices = self.scan_slices(flat_indices, direction)

            for slice_idx, c_slice in enumerate(direction_slices):
                slice_indices = np.unravel_index(
                    np.squeeze(direction_flat_indices[slice_idx], axis=0),
                    (num_of_rows, num_of_cols)
                )

                l_slice = self.dp_grade_slice(c_slice, p1, p2).T 
                l_direction_tensors[i][slice_indices] = l_slice

        l = np.mean(l_direction_tensors, axis=0)

        return self.naive_labeling(l)
