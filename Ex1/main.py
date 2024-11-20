import time
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from cv2 import resize, INTER_CUBIC
from matplotlib.patches import Circle

from ex1_student_solution import Solution


##########################################################
# Don't forget to fill in your IDs!!!
# students' IDs:
ID1 = '206869000'
ID2 = '209951383'
##########################################################


def tic():
    return time.time()


def toc(t):
    return float(tic()) - float(t)


def load_data(is_perfect_matches=True):
    # Read the data:
    src_img = mpimg.imread('src.jpg')
    dst_img = mpimg.imread('dst.jpg')
    if is_perfect_matches:
        # loading perfect matches
        matches = scipy.io.loadmat('matches_perfect')
    else:
        # matching points and some outliers
        matches = scipy.io.loadmat('matches')
    match_p_dst = matches['match_p_dst'].astype(float)
    match_p_src = matches['match_p_src'].astype(float)
    return src_img, dst_img, match_p_src, match_p_dst


def main():
    solution = Solution()
    # Parameters
    max_err = 25
    inliers_percent = 0.8
    # loading data with perfect matches
    src_img, dst_img, match_p_src, match_p_dst = load_data()
    # Compute naive homography
    tt = time.time()
    naive_homography = solution.compute_homography_naive(match_p_src,
                                                         match_p_dst)
    plt.figure()
    plt.imshow(src_img)

    print('Naive Homography {:5.4f} sec'.format(toc(tt)))
    print(naive_homography)

    # Plot naive homography with forward mapping, slow implementation
    tt = time.time()
    transformed_image = solution.compute_forward_homography_slow(
        homography=naive_homography,
        src_image=src_img,
        dst_image_shape=dst_img.shape)

    print('Naive Homography Slow computation takes {:5.4f} sec'.format(toc(tt)))
    plt.figure()
    plt.imshow(transformed_image)
    plt.title('Forward Homography Slow implementation')
    plt.savefig('outputs/slow_naive_homography.jpeg')
    plt.show()

    # Plot naive homography with forward mapping, fast implementation
    tt = time.time()
    transformed_image_fast = solution.compute_forward_homography_fast(
        homography=naive_homography,
        src_image=src_img,
        dst_image_shape=dst_img.shape)
    print('Naive Homography Fast computation takes {:5.4f} sec'.format(toc(tt)))
    plt.figure()
    forward_panorama_fast_plot = plt.imshow(transformed_image_fast)
    plt.title('Forward Homography Fast implementation')
    # plt.show()
    plt.savefig('outputs/fast_naive_homography.jpeg')

    # loading data with imperfect matches
    src_img, dst_img, match_p_src, match_p_dst = load_data(False)

    # Compute naive homography
    tt = time.time()
    naive_homography = solution.compute_homography_naive(match_p_src,
                                                         match_p_dst)
    print('Naive Homography for imperfect matches {:5.4f} sec'.format(toc(tt)))
    print(naive_homography)

    # Plot naive homography with forward mapping, fast implementation for
    # imperfect matches
    tt = time.time()
    transformed_image_fast = solution.compute_forward_homography_fast(
        homography=naive_homography,
        src_image=src_img,
        dst_image_shape=dst_img.shape)

    print('Naive Homography Fast computation for imperfect matches takes '
          '{:5.4f} sec'.format(toc(tt)))
    plt.figure()
    forward_panorama_imperfect_matches_plot = plt.imshow(transformed_image_fast)
    plt.title('Forward Panorama imperfect matches')
    # plt.show()
    plt.savefig('outputs/fast_naive_homography_imperfect.jpeg')

    # Test naive homography
    tt = time.time()
    fit_percent, dist_mse = solution.test_homography(naive_homography,
                                                     match_p_src,
                                                     match_p_dst,
                                                     max_err)
    print('Naive Homography Test {:5.4f} sec'.format(toc(tt)))
    print([fit_percent, dist_mse])

    # Compute RANSAC homography
    tt = tic()
    ransac_homography = solution.compute_homography(match_p_src,
                                                    match_p_dst,
                                                    inliers_percent,
                                                    max_err)
    print('RANSAC Homography {:5.4f} sec'.format(toc(tt)))
    print(ransac_homography)

    # Test RANSAC homography
    tt = tic()
    fit_percent, dist_mse = solution.test_homography(ransac_homography,
                                                     match_p_src,
                                                     match_p_dst,
                                                     max_err)
    print('RANSAC Homography Test {:5.4f} sec'.format(toc(tt)))
    print([fit_percent, dist_mse])

    tt = time.time()
    transformed_image_fast = solution.compute_forward_homography_fast(
        homography=ransac_homography,
        src_image=src_img,
        dst_image_shape=dst_img.shape)
    print('RANSAC Homography Fast computation for imperfect matches takes '
          '{:5.4f} sec'.format(toc(tt)))
    plt.figure()
    forward_panorama_imperfect_matches_plot = plt.imshow(transformed_image_fast)
    plt.title('Forward RANSAC Panorama imperfect matches')
    plt.show()
    plt.savefig('outputs/fast_ransac_homography_imperfect.jpeg')

    # Build backward warp
    backward_homography = solution.compute_homography(match_p_dst, 
                                                      match_p_src,
                                                      inliers_percent,
                                                      max_err)
    tt = time.time()
    backward_warp_image = solution.compute_backward_mapping(
                                      backward_projective_homography=backward_homography,
                                      src_image=src_img,
                                      dst_image_shape=dst_img.shape)
    print('Backward Warp takes {:5.4f} sec'.format(toc(tt)))
    plt.figure()
    course_backward_warp_img = plt.imshow(backward_warp_image.astype(np.uint8))
    plt.title('Backward warp example')
    plt.savefig('outputs/backward_warp_imperfect.jpeg')
    # plt.show()

    # Build panorama
    tt = tic()
    img_pan = solution.panorama(src_img,
                                dst_img,
                                match_p_src,
                                match_p_dst,
                                inliers_percent,
                                max_err)
    print('Panorama {:5.4f} sec'.format(toc(tt)))

    # Course panorama
    plt.figure()
    course_panorama_plot = plt.imshow(img_pan)
    plt.title('Great Panorama')
    # plt.show()
    plt.savefig('outputs/panorama.jpg')


def your_images_loader():
    src_img_test = mpimg.imread('src_test.jpg')
    dst_img_test = mpimg.imread('dst_test.jpg')

    DECIMATION_FACTOR = 5.0
    src_img_test = resize(src_img_test,
                          dsize=(int(src_img_test.shape[1]/DECIMATION_FACTOR),
                                 int(src_img_test.shape[0]/DECIMATION_FACTOR)),
                          interpolation=INTER_CUBIC)
    dst_img_test = resize(dst_img_test,
                          dsize=(int(dst_img_test.shape[1]/DECIMATION_FACTOR),
                                 int(dst_img_test.shape[0]/DECIMATION_FACTOR)),
                          interpolation=INTER_CUBIC)

    matches_test = scipy.io.loadmat('matches_test')

    match_p_dst = matches_test['match_p_dst'].astype(float)
    match_p_src = matches_test['match_p_src'].astype(float)

    match_p_dst /= DECIMATION_FACTOR
    match_p_src /= DECIMATION_FACTOR
    return src_img_test, dst_img_test, match_p_src, match_p_dst


def your_images_main():
    solution = Solution()
    # Student Files
    # first run "create_matching_points.py" with your own images to create a mat
    # file with the matching coordinates.
    max_err = 25  # <<<<< YOU MAY CHANGE THIS
    inliers_percent = 0.8  # <<<<< YOU MAY CHANGE THIS

    src_img_test, dst_img_test, match_p_src, match_p_dst = your_images_loader()

    # scatter the matching points
    plt.figure()
    plt.imshow(src_img_test)
    plt.scatter(*match_p_src, edgecolors='blue', facecolors='none', s=20)
    plt.title('src matching points')
    plt.show()
    plt.savefig('outputs/test_src_scattered.jpg')

    plt.figure()
    plt.imshow(dst_img_test)
    plt.scatter(*match_p_dst, edgecolors='red', facecolors='none', s=20)
    plt.title('dst matching points')
    plt.show()
    plt.savefig('outputs/test_dst_scattered.jpg')

    homography = solution.compute_homography(match_p_src,
                                             match_p_dst,
                                             inliers_percent,
                                             max_err)
    # Test student homography
    tt = time.time()
    fit_percent, dist_mse = solution.test_homography(homography,
                                                     match_p_src,
                                                     match_p_dst,
                                                     max_err)
    print('Student Homography Test {:5.4f} sec'.format(toc(tt)))
    print([fit_percent, dist_mse])

    img = solution.compute_forward_homography_fast(
        homography=homography,
        src_image=src_img_test,
        dst_image_shape=dst_img_test.shape)
    plt.figure()
    student_forward_warp_img = plt.imshow(img.astype(np.uint8))
    plt.title('Forward warp example')
    # plt.show()
    plt.savefig('outputs/test_forward_homography.jpg')

    backward_homography = solution.compute_homography(match_p_dst, match_p_src,
                                                      inliers_percent,
                                                      max_err=25)
    img = solution.compute_backward_mapping(
        backward_projective_homography=backward_homography,
                                      src_image=src_img_test,
                                      dst_image_shape=dst_img_test.shape)
    plt.figure()
    student_backward_warp_img = plt.imshow(img.astype(np.uint8))
    plt.title('Backward warp example')
    # plt.show()
    plt.savefig('outputs/test_backward_homography.jpg')

    # Build student panorama
    tt = tic()
    img_pan = solution.panorama(src_img_test, dst_img_test, match_p_src,
                                match_p_dst, inliers_percent, max_err)
    print('Student Panorama {:5.4f} sec'.format(toc(tt)))

    plt.figure()
    student_panorama = plt.imshow(img_pan)
    plt.title('Awesome Panorama')
    # plt.show()
    plt.savefig('outputs/test_panorama.jpg')

    # Build reversed student panorama
    tt = tic()
    img_pan2 = solution.panorama(dst_img_test, src_img_test, match_p_dst,
                                 match_p_src, inliers_percent, max_err)
    print('Student Panorama {:5.4f} sec'.format(toc(tt)))

    plt.figure()
    reversed_student_panorama = plt.imshow(img_pan2)
    plt.title('Reversed Awesome Panorama')
    # plt.show()
    plt.savefig('outputs/test_panorama_2.jpg')


if __name__ == '__main__':
    main()
    your_images_main()
