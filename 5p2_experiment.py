import multiprocessing as mp
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from mrf_cut import segment_image
from bar_example import create_bar, bar_generator, add_gray_noise, add_flip_noise

"""
Run the experiment, return the losses
"""


def experiment_gray(b, num_trials=100):
    bar_ref = create_bar()

    def loss(pred_image):
        return 1 - np.mean(pred_image == bar_ref)

    def pairwise(val1, val2, idx1, idx2):
        return b

    losses = []
    for i in range(num_trials):
        noisy = add_gray_noise(bar_ref)
        map_bar = segment_image(noisy, pairwise_potential_fnc=pairwise)
        losses.append(loss(map_bar))
    return np.array(losses).mean()


def experiment_binary(b, perc=0.1, num_trials=100):
    bar_ref = create_bar()

    def loss(pred_image):
        return 1 - np.mean(pred_image == bar_ref)

    def loss(pred_image):
        return max(
            np.mean(pred_image[bar_ref == 0] == 1),
            np.mean(pred_image[bar_ref == 1] == 0),
        )

    # def loss(pred_image):
    #     return 1 - (
    #         np.sum(pred_image * bar_ref) / np.sum(np.minimum(pred_image + bar_ref, 1))
    #     )

    def pairwise(val1, val2, idx1, idx2):
        return b

    losses = []
    for i in range(num_trials):
        noisy = add_flip_noise(bar_ref, perc=perc)

        noisy_mean = np.mean(noisy)

        def unary(val, idx):
            if val < noisy_mean:
                return 1 - noisy_mean, noisy_mean
            return noisy_mean, 1 - noisy_mean

        def unary(val, *args):
            # val = bar_ref[idx]
            p = 0.5
            bp = perc
            return val * bp, (1 - val) * (1 - bp)
            if val > noisy_mean:
                return 1 - noisy_mean * p, noisy_mean * p
            return noisy_mean * p, 1 - noisy_mean * p

        map_bar = segment_image(
            noisy, unary_potential_fnc=unary, pairwise_potential_fnc=pairwise
        )
        losses.append(loss(map_bar))
    return np.array(losses).mean()


if __name__ == "__main__":
    betas = np.linspace(0.1, 4, 8)
    percs = np.linspace(0.05, 0.5, 12)

    result_array = []
    with mp.Pool(4) as pool:
        for beta in betas:
            results = pool.map(partial(experiment_binary, beta, num_trials=100), percs)
            result_array.append(results)
    result_array = np.array(result_array)
    np.save("output/5p2_results", result_array)