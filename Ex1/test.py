import scipy.io
import matplotlib.image as mpimg

from ex1_student_solution import Solution

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

src_img, dst_img, match_p_src, match_p_dst = load_data(False)

Solution().compute_homography(match_p_src, match_p_dst, 0.8, 25)
