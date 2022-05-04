from numba import njit
import numpy as np
@njit(parallel=True)
def resixe_3d(img):
    l = np.random.randint(0, 255, (3, 460, 322))
    for a in range(len(img)):
        for b in range(len(img[a])):
            for c in range(len(img[a][b])):
                l[c][a][b] = img[a][b][c]
    return l

@njit(parallel=True)
def rech(frame, sdvik, size, tol, video_h, video_w):
    for j in range(sdvik, sdvik + video_w):
        for i in range(tol):
            frame[i][j] = [0, 255, 0]
            frame[i + video_h - tol][j] = [0, 255, 0]
            frame[j - sdvik][i + sdvik] = [0, 255, 0]
            frame[j - sdvik - 1][i + sdvik + video_w] = [0, 255, 0]
    for j in range(video_w, video_h):
        for i in range(tol):
            frame[j][i + sdvik] = [0, 255, 0]
            frame[j][i + sdvik + video_w] = [0, 255, 0]
    for i in range(size[1]):
        for j in range(size[0] // 2 - 1):
            frame[i][j][0], frame[i][size[0] - 1 - j][0] = frame[i][size[0] - 1 - j][0], frame[i][j][0]
            frame[i][j][1], frame[i][size[0] - 1 - j][1] = frame[i][size[0] - 1 - j][1], frame[i][j][1]
            frame[i][j][2], frame[i][size[0] - 1 - j][2] = frame[i][size[0] - 1 - j][2], frame[i][j][2]

@njit(parallel=True)
def rech_new(frame, size):
    for i in range(size[1]):
        for j in range(size[0] // 2 - 1):
            frame[i][j][0], frame[i][size[0] - 1 - j][0] = frame[i][size[0] - 1 - j][0], frame[i][j][0]
            frame[i][j][1], frame[i][size[0] - 1 - j][1] = frame[i][size[0] - 1 - j][1], frame[i][j][1]
            frame[i][j][2], frame[i][size[0] - 1 - j][2] = frame[i][size[0] - 1 - j][2], frame[i][j][2]
