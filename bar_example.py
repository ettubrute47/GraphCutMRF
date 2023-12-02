import numpy as np


def px(px, M, N):
    return int(px * N / M)


def create_bar(offset=0):
    N = 32
    M = 32

    def px(x):
        return int(x * N / M)

    bar = np.ones((N, N))

    bar[px(4 + offset) : px(-4 + offset), px(14) : px(-14)] = 0
    bar[px(18 + offset) : px(22 + offset), px(14) : px(-14)] = 1
    bar[px(18 + offset) : px(20 + offset), px(16) : px(-14)] = 0
    bar[px(4), px(14)] = 1  # or comment in the noise
    return bar


def add_gray_noise(clean, scale=0.25):
    N = 32
    M = 32
    bar = np.array(
        clean
    )  # is 0, 1, want to have if < 0.5 still likely to be 0, and if > 0.5 still likely to be 1
    noise = np.random.normal(scale=scale, size=(N, N))
    bar[bar < 0.5] += np.abs(noise[bar < 0.5])
    bar[bar >= 0.5] -= np.abs(noise[bar >= 0.5])
    return np.clip(bar, 0, 1)


def add_flip_noise(clean, perc=0.1):
    N = 32
    M = 32
    bar = np.array(
        clean
    )  # is 0, 1, want to have if < 0.5 still likely to be 0, and if > 0.5 still likely to be 1
    noise_mask = np.random.rand(N, M) < perc
    bar[noise_mask] = 1 - bar[noise_mask]
    return np.clip(bar, 0, 1)


def bar_generator():
    i = 0
    offset = 0
    max_offset = 4
    direction = 1
    while True:
        yield create_bar(offset)
        i += 1
        if i % max_offset == 0:
            direction *= -1
            i = 0
        offset += direction
