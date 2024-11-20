import scipy.io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from ex1_student_solution import Solution

def load_data():
    # Read the data:
    src_img = mpimg.imread('src_test.jpg')
    dst_img = mpimg.imread('dst_test.jpg')
    matches = scipy.io.loadmat('matches_test')
    match_p_dst = matches['match_p_dst'].astype(float)
    match_p_src = matches['match_p_src'].astype(float)
    return src_img, dst_img, match_p_src, match_p_dst

src_img, dst_img, match_p_src, match_p_dst = load_data()

plt.figure()
plt.imshow(src_img)
plt.scatter(*match_p_src, edgecolors='blue', facecolors='none', s=20)
plt.title('src matching points')
plt.show()
plt.savefig('outputs/test_src_scattered.jpg')

plt.figure()
plt.imshow(dst_img)
plt.scatter(*match_p_dst, edgecolors='red', facecolors='none', s=20)
plt.title('dst matching points')
plt.show()
plt.savefig('outputs/test_dst_scattered.jpg')