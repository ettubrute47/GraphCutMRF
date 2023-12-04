import numpy as np


def px(px, M, N):
    return int(px * N / M)


def create_bar(xoffset=0, yoffset=0, height=1):
    N = 32
    M = 32

    def px(x):
        return int(x * N / M)

    bar = np.ones((N, N))

    dh = 4
    dh2 = 14
    bar[px(dh + yoffset) : px(-dh + yoffset), px(14 + xoffset) : px(-14 + xoffset)] = 0
    bar[
        px(dh + dh2 + yoffset) : px(dh + dh2 + dh + yoffset),
        px(14 + xoffset) : px(-14 + xoffset),
    ] = 1
    bar[
        px(dh + dh2 + yoffset) : px(dh + dh2 + dh / 2 + yoffset),
        px(16 + xoffset) : px(-14 + xoffset),
    ] = 0
    bar[px(4), px(14)] = 1  # or comment in the noise
    return bar


def create_face(*args):
    N = 32
    bar = np.ones((N, N))
    ball_center = int(N / 3)
    center = int(N / 2)
    dr = int(N * 0.17) ** 2
    spacing = int(N / 3.5)
    bc1 = [ball_center, center - spacing]
    bc2 = [ball_center, center + spacing]
    bar_center_x = N - ball_center  # x
    bar_length = int(N * 0.4)
    bar_height = int(N * 0.1)

    hairline = int(N * 0.07)
    for i in range(N):
        for j in range(N):
            dr1_ij = (bc1[0] - i) ** 2 + (bc1[1] - j) ** 2
            dr2_ij = (bc2[0] - i) ** 2 + (bc2[1] - j) ** 2
            # then the bar at
            if dr1_ij < dr or dr2_ij < dr:
                bar[i, j] = 0

            if abs(bar_center_x - i) < bar_height and abs(center - j) < bar_length:
                bar[i, j] = 0

            if i < hairline or i >= N - hairline:
                bar[i, j] = 0

            if abs(i - center) < int(N * 0.08) and abs(j - center) < int(N * 0.08):
                bar[i, j] = 0
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.imshow(create_bar())
    print(np.mean(create_bar() == 1))
    plt.show()
