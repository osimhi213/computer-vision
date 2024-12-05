import numpy as np

from solution import Solution

solution = Solution()

shape = (4, 5)
left_image = np.arange(1, 21).reshape(shape).astype(np.double)
right_image = np.zeros(shape, dtype=np.double)
for i in range(len(right_image)):
    right_image[i, :] = i + 1

win_size = 3
d = win_size // 2
ssd_tensor = solution.ssd_distance(left_image, right_image, win_size, 2)
print(ssd_tensor[1, 2, 1])
