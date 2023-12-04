import multiprocessing as mp
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from mrf_cut import segment_image
from mrf_cut5 import segment_image_igraph
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

    def balanced_loss(pred_image):
        return 0.5 * np.mean(pred_image[bar_ref == 0] == 1) + 0.5 * np.mean(
            pred_image[bar_ref == 1] == 0
        )

    def loss(pred_image, mle_image):
        acc_map = balanced_loss(pred_image)
        acc_mle = balanced_loss(mle_image)
        # score from 0 to 1, where 1 is 100% accurate,
        return [acc_mle, acc_map]

    # def loss(pred_image):
    #     return 1 - (
    #         np.sum(pred_image * bar_ref) / np.sum(np.minimum(pred_image + bar_ref, 1))
    #     )

    def pairwise(val1, val2, idx1, idx2):
        return b

    # x cols, a rows
    dens = np.mean(bar_ref == 1)
    joint_table = [
        [dens * (1 - perc), (1 - dens) * perc],
        [dens * perc, (1 - dens) * (1 - perc)],
    ]
    losses = []
    for i in range(num_trials):
        noisy = add_flip_noise(bar_ref, perc=perc)

        noisy_mean = 1 - np.mean(bar_ref)
        # noisy_mean = 1 - noisy_mean
        if num_trials == 1:
            print(noisy_mean)

        def unary(val, idx):
            if val < noisy_mean:
                return 1 - noisy_mean, noisy_mean
            return noisy_mean, 1 - noisy_mean

        def unary(val, *args):
            # returns p(a|1), p(a|0)
            # val = bar_ref[idx]
            # F(y|0) = val == 0
            p = 0.5
            bp = noisy_mean
            bp = noisy_mean * (1 - perc) + (1 - noisy_mean) * perc
            bp = noisy_mean * (1 - perc) + (1 - noisy_mean) * perc
            bp = perc
            proba_1 = bp ** (1 - val) * (1 - bp) ** val
            return val * perc, (1 - val) * (1 - perc)
            if not np.isscalar(val):
                pos = []
                neg = []
                for va in val.flat:
                    p, n = unary(va)
                    pos.append(p)
                    neg.append(n)
                return np.array(pos).reshape(val.shape), np.array(neg).reshape(
                    val.shape
                )
                return np.array(list(map(unary, val.flat))).reshape((2, val.size))
            val = int(val)
            proba_val = joint_table[val][0] + joint_table[val][1]
            p_1 = joint_table[val][0] + joint_table[1 - val][0]
            p_2 = joint_table[val][1] + joint_table[1 - val][1]

            # return joint_table[val][0] / p_1, joint_table[val][1] / p_2
            return joint_table[val][1] / p_2, joint_table[val][0] / p_1
            # return val * perc, (1 - val) * (1 - perc)

            return val * 0.8, (1 - val) * 0.8
            return val * bp, (1 - val) * (1 - bp)
            return val * (1 - perc) * noisy_mean, (1 - val) * (1 - perc) * (
                1 - noisy_mean
            )
            return proba_1, (1 - proba_1)
            if val > noisy_mean:
                return 1 - noisy_mean * p, noisy_mean * p
            return noisy_mean * p, 1 - noisy_mean * p

        # def unary(image):
        #     unary_potentials = np.zeros((image.size, 2), dtype=np.float32)
        #     unary_potentials[image.flat == 1, 0] = perc
        #     unary_potentials[image.flat == 1, 1] = 1 - perc
        #     unary_potentials[image.flat == 0, 0] = 1 - perc
        #     unary_potentials[image.flat == 0, 1] = perc
        #     return 1 - unary_potentials.T

        # def unary(image):
        #     unary_potentials = np.zeros((image.size, 2), dtype=np.float32)
        #     unary_potentials[:, 0] = image.flatten() * perc
        #     unary_potentials[:, 1] = (1 - image.flatten()) * (1 - perc)
        #     return 1 - unary_potentials.T

        map_bar = segment_image(
            noisy, unary_potential_fnc=unary, pairwise_potential_fnc=pairwise
        )
        mb2 = np.ones(map_bar.shape)
        mb2[:, :-2] = map_bar[:, 2:]
        # map_bar = mb2
        if num_trials == 1:
            print(
                "Loss: ",
                loss(map_bar, noisy),
                " vs ",
                loss(noisy, noisy),
                " vs ",
                loss(mb2, noisy),
                " for perc ",
                perc,
            )
            fig, axs = plt.subplots(1, 5)
            pos, neg = unary(noisy)
            axs[0].imshow(pos.reshape(noisy.shape), vmin=0, vmax=1)
            axs[1].imshow(neg.reshape(noisy.shape), vmin=0, vmax=1)
            axs[2].imshow(map_bar, vmin=0, vmax=1)
            axs[4].imshow(noisy, vmin=0, vmax=1)
            obj = axs[3].imshow(noisy * 0.2 + map_bar * 0.8, vmin=0, vmax=1)
            plt.colorbar(obj)
            plt.show()
        losses.append(loss(map_bar, noisy))
    return np.array(losses)


if __name__ == "__main__":
    betas = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])  # np.linspace(0.1, 2, 8)
    betas = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])  # np.linspace(0.1, 2, 8)
    percs = np.linspace(0.05, 0.5, 12)

    experiment_binary(4, perc=0.2, num_trials=1)

    result_array = []
    with mp.Pool(6) as pool:
        for beta in betas:
            print("Beta ", beta)
            results = pool.map(partial(experiment_binary, beta, num_trials=100), percs)
            result_array.append(results)
    result_array = np.array(result_array)
    np.save("output/5p2_results", result_array)
