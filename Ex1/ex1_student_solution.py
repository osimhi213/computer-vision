"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        # return homography
        """INSERT YOUR CODE HERE"""
        N = match_p_src.shape[1]
        if N < 4:
            raise ValueError('Error - the number of matching point is insufficient. {0} < 4'.format(N))
   
        # Construct A
        A = np.zeros((2 * N, 9))
        j = 0
        for i in range(N):
            j = 2 * i
            X_i = np.append(match_p_src[:, i], 1)
            X_itag = np.append(match_p_dst[:, i], 1)
            A[j, :] = np.concatenate((X_i.T, [0, 0, 0], -1 * X_itag[0] * X_i.T))
            A[j + 1, :] = np.concatenate(([0, 0, 0], X_i.T, -1 * X_itag[1] * X_i.T))

        # SVD
        eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T, A))
        # The homography vector is the eigenvector corresponding to the smallest eigenvalue
        h = eigenvectors[:, np.argmin(np.abs(eigenvalues))]
        H = h.reshape(3, 3)

        return H

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""
        dst_image = np.zeros(dst_image_shape, dtype=src_image.dtype)
        h1, w1 = src_image.shape[:-1]
        h2, w2 = dst_image.shape[:-1]

        for v in range(h1):
            for u in range(w1):
                x = np.array([u, v, 1])
                x_tag = np.dot(homography, x)
                x_tag = x_tag / x_tag[-1]
                u_tag, v_tag = np.round(x_tag[0:2]).astype(np.int) # Nearest neighboor
                if (0 <= u_tag < w2 and 0 <= v_tag < h2):
                    dst_image[v_tag, u_tag, :] = src_image[v, u, :]

        return dst_image

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""
        dst_image = np.zeros(dst_image_shape, dtype=src_image.dtype)
        h1, w1 = src_image.shape[:-1]
        h2, w2 = dst_image.shape[:-1]

        # 1
        U, V = np.meshgrid(np.arange(w1), np.arange(h1))
        u = U.reshape(1, -1)
        v = V.reshape(1, -1)
        z = np.ones((1, h1 * w1))
        # 2
        P = np.concatenate((u, v, z))
        # 3
        P_tag = np.dot(homography, P)
        # 4
        P_tag = P_tag / P_tag[-1, :]
        U_tag, V_tag = np.round(P_tag[0:2]).astype(np.int)
        valid_coords = np.where((U_tag >= 0) & (U_tag < w2) & (V_tag >= 0) & (V_tag < h2))[0]
        u = u.reshape(-1)[valid_coords]
        v = v.reshape(-1)[valid_coords]
        u_tag = U_tag[valid_coords]
        v_tag = V_tag[valid_coords]
        # 5
        dst_image[v_tag, u_tag, :] = src_image[v, u, :]

        return dst_image


    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        # return fit_percent, dist_mse
        """INSERT YOUR CODE HERE"""
        N = match_p_src.shape[1]
        z = np.ones((1, N), dtype=match_p_src.dtype)
        x = np.concatenate((match_p_src, z), axis=0)
        x_tag_pred = np.dot(homography, x)
        mapped_p_dest = np.round(x_tag_pred / x_tag_pred[-1])[:2].astype(np.int)

        dists = np.linalg.norm(mapped_p_dest - match_p_dst, ord=2, axis=0)
        inliers = dists <= max_err
        num_of_inliers = np.count_nonzero(inliers)
        fit_precent = num_of_inliers / N

        if num_of_inliers > 0:
            inliers_errors = np.linalg.norm((mapped_p_dest - match_p_dst)[:, inliers], ord=2, axis=0)
            dist_mse = np.mean(inliers_errors ** 2)
        else:
            dist_mse = 1e9

        return fit_precent, dist_mse

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        # return mp_src_meets_model, mp_dst_meets_model
        """INSERT YOUR CODE HERE"""
        N = match_p_src.shape[1]
        z = np.ones((1, N), dtype=match_p_src.dtype)
        x = np.concatenate((match_p_src, z), axis=0)
        x_tag_pred = np.dot(homography, x)
        mapped_p_dest = np.round(x_tag_pred / x_tag_pred[-1])[:2].astype(np.int)

        errors = np.linalg.norm(mapped_p_dest - match_p_dst, ord=2, axis=0)
        inliers = errors <= max_err
        
        mp_src_meets_model = match_p_src[:, inliers]
        mp_dst_meets_model = match_p_dst[:, inliers]

        return mp_src_meets_model, mp_dst_meets_model

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # use class notations:
        w = inliers_percent
        t = max_err
        # p = parameter determining the probability of the algorithm to
        # succeed
        p = 0.99
        # the minimal probability of points which meets with the model
        d = 0.5
        # number of points sufficient to compute the model
        n = 4
        # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        # return homography
        """INSERT YOUR CODE HERE"""
        N = match_p_src.shape[1]
        optimal_dist_mse = np.inf
        optimal_homography = None
        for i in range(k):
            points_set = np.random.choice(range(N), size=n, replace=False)
            H = self.compute_homography_naive(match_p_src[:, points_set], match_p_dst[:, points_set])
            fit_precent, _ = self.test_homography(H, match_p_src, match_p_dst, t)
            if fit_precent >= d:
                mp_src_meets_model, mp_dst_meets_model = self.meet_the_model_points(H, match_p_src, match_p_dst, t)
                H = self.compute_homography_naive(mp_src_meets_model, mp_dst_meets_model)
                _, dist_mse = self.test_homography(H, match_p_src, match_p_dst, t)                    

                if dist_mse < optimal_dist_mse:
                    optimal_homography = H
                    optimal_dist_mse = dist_mse

        return optimal_homography

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # return backward_warp
        dst_image = np.zeros(dst_image_shape, dtype=src_image.dtype)
        h1, w1, num_chanels = dst_image.shape
        h2, w2 = src_image.shape[:-1]

        # 1
        U, V = np.meshgrid(np.arange(w1), np.arange(h1))
        u = U.reshape(1, -1)
        v = V.reshape(1, -1)
        z = np.ones((1, h1 * w1))
        # 2
        P = np.concatenate((u, v, z))
        # 3
        P_tag = np.dot(backward_projective_homography, P)
        P_tag /= P_tag[-1, :]
        U_tag, V_tag = P_tag[0:2]
        valid_coords = np.where((U_tag >= 0) & (U_tag < w2) & (V_tag >= 0) & (V_tag < h2))[0]
        u = u.reshape(-1)[valid_coords]
        v = v.reshape(-1)[valid_coords]
        u_tag = U_tag[valid_coords]
        v_tag = V_tag[valid_coords]
        # 4
        src_x_points, src_y_points = np.meshgrid(np.arange(w2), np.arange(h2))
        src_x_points = src_x_points.reshape(-1)
        src_y_points = src_y_points.reshape(-1)
        # 5
        for ch in range(num_chanels):
            dst_image[v, u, ch] = griddata((src_x_points, src_y_points), src_image[:, :, ch].reshape(-1), (u_tag, v_tag), method='cubic')

        return dst_image

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # return final_homography
        """INSERT YOUR CODE HERE"""
        T = np.array([[1, 0 , -pad_left], [0, 1, -pad_up], [0, 0, 1]])
        H = np.dot(backward_homography, T)
        H /= np.linalg.det(H)

        return H

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # return np.clip(img_panorama, 0, 255).astype(np.uint8)
        """INSERT YOUR CODE HERE"""
        # 1
        forward_homography = self.compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)
        panorama_rows_num, panorama_cols_num, pad_struct = self.find_panorama_shape(src_image, dst_image, forward_homography)
        panorama_shape = (panorama_rows_num, panorama_cols_num, dst_image.shape[-1])

        # 2
        backward_homography = self.compute_homography(match_p_dst, match_p_src, inliers_percent, max_err)

        # 3
        backward_homography = self.add_translation_to_backward_homography(backward_homography, pad_struct.pad_left, pad_struct.pad_up)

        # 4
        bacward_warp = self.compute_backward_mapping(backward_homography, src_image, panorama_shape)

        # 5
        panorama = np.zeros(panorama_shape, dtype=dst_image.dtype)
        h2, w2, _ = dst_image.shape
        pl = pad_struct.pad_left
        pu = pad_struct.pad_up
        panorama[pu: h2 + pu, pl:w2 + pl, :] = dst_image

        # 6
        indices = panorama == 0
        panorama[indices] = bacward_warp[indices]

        # 7
        return np.clip(panorama, 0, 255).astype(dst_image.dtype)
