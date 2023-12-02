import tkinter as tk
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image, ImageTk, ImageDraw
import matplotlib

from scipy import stats
from scipy.ndimage import label, sum as ndi_sum

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from mrf_cut import segment_image


def create_distance_array(image_shape, draw_history):
    y_indices, x_indices = np.indices(image_shape)
    all_points = np.stack((x_indices, y_indices), axis=-1).reshape(-1, 2)

    draw_history_array = np.array(draw_history)
    distances = cdist(all_points, draw_history_array)

    # Find the minimum distance to draw_history for each point
    min_distances = distances.min(axis=1)
    distance_array = min_distances.reshape(image_shape)

    return distance_array


class ImageDrawApp:
    def __init__(
        self,
        image_path,
        primary_color="red",
        secondary_color="green",
        max_size=(200, 200),
    ):
        self.original_image = Image.open(image_path)
        self.image = self.resize_image(self.original_image, max_size)
        self.image = self.image.convert("L").convert("RGBA")
        self.image_shape = self.image.size[::-1]
        self.current_color = primary_color
        self.primary_color = primary_color
        self.mesh_color = (0, 0, 255, 120)

        self.secondary_color = secondary_color
        self.overlay = Image.new("RGBA", self.image.size)
        self.colors = [primary_color, secondary_color]
        self.draw_mode = 0

        self.draw_history = []

        self.root = tk.Tk()
        self.root.title("Draw on Image App")

        self.tk_image = ImageTk.PhotoImage(self.image)

        self.canvas = tk.Canvas(
            self.root, width=self.tk_image.width(), height=self.tk_image.height()
        )
        self.canvas.pack(side="left", fill="both", expand="yes")

        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        self.draw = ImageDraw.Draw(self.image)

        self.draw_distance = np.ones(self.image_shape, int)
        self.mask = np.zeros(self.image_shape, int)
        self.set_overlay_mask(self.mask)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", lambda *args: self.update_canvas())

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.subplot = self.figure.add_subplot(111)

        self.plot_canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.plot_widget = self.plot_canvas.get_tk_widget()
        self.plot_widget.pack(side="right", fill="both", expand=True)
        self.initialize_plot()

        self.color_button = tk.Button(
            self.root, text="Change Color", command=self.toggle_color
        )
        self.color_button.pack()

        self.plot_button = tk.Button(
            self.root, text="Update Plot", command=self.update_plot
        )
        self.plot_button.pack()

        self.root.mainloop()

    def paint(self, event):
        radius = 1
        x1, y1 = (event.x - radius), (event.y - radius)
        x2, y2 = (event.x + radius), (event.y + radius)
        self.draw_history.append((self.draw_mode, x1, y1, x2, y2))
        self.canvas.create_oval(
            x1, y1, x2, y2, fill=self.current_color, outline=self.current_color
        )
        self.draw.ellipse(
            [x1, y1, x2, y2], fill=self.current_color, outline=self.current_color
        )
        self.draw_distance = create_distance_array(
            self.image_shape, [[d[1], d[2]] for d in self.draw_history if d[0] == 0]
        )
        # I want to get the intensities for that too
        # self.set_overlay_mask(mask)
        # self.update_plot()
        # the unary potential will just be the distance from the guy and not...

    # def reset(self, event):
    #     self.tk_image = ImageTk.PhotoImage(self.image)
    #     self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def toggle_color(self):
        self.draw_mode = (self.draw_mode + 1) % 2
        self.current_color = self.colors[self.draw_mode]

    def update_plot1(self, *args):
        grayscale_image = np.array(self.image.convert("L"))
        foreground_intensities = grayscale_image[self.mask == 1]
        background_intensities = grayscale_image[self.mask == 0]
        foreground_kde = stats.gaussian_kde(foreground_intensities)
        background_kde = stats.gaussian_kde(background_intensities)

        histogram = np.histogram(grayscale_image, bins=256, range=(0, 256))[0]
        histogram = histogram.astype(float) / histogram.sum()
        self.subplot.clear()
        self.subplot.plot(histogram)
        if np.any(self.mask == 1):
            histogram_white = np.histogram(
                grayscale_image[self.mask == 1], bins=256, range=(0, 256)
            )[0]
            histogram_white = histogram_white.astype(float) / histogram_white.sum()
            self.subplot.plot(histogram_white)

            histogram_black = np.histogram(
                grayscale_image[self.mask == 0], bins=256, range=(0, 256)
            )[0]
            histogram_black = histogram_black.astype(float) / histogram_black.sum()
            self.subplot.plot(histogram_black)
        self.subplot.set_title("Intensity distributions")
        self.plot_canvas.draw()

    def update_plot(self, *args):
        distance_mask = self.draw_distance < 5
        if not np.any(distance_mask == 1):
            return
        grayscale_image = np.array(self.image.convert("L"))
        foreground_mask = (distance_mask == 1) | (self.mask == 1)
        foreground_intensities = grayscale_image[foreground_mask]
        background_intensities = grayscale_image[~foreground_mask]
        print("Calculating kde from plot")
        foreground_kde = stats.gaussian_kde(foreground_intensities)
        print("Calculating kde from plot 2")
        background_kde = stats.gaussian_kde(background_intensities)
        self.subplot.clear()
        ming = grayscale_image.min()
        maxg = grayscale_image.max()
        intens = np.linspace(ming, maxg)
        foreground_pdf = foreground_kde(intens)
        background_pdf = background_kde(intens)
        self.subplot.plot(foreground_pdf)
        self.subplot.plot(background_pdf)
        # self.subplot.plot(background_kde(intens))
        self.subplot.set_title("Intensity distributions")
        self.plot_canvas.draw()

        def unary(val, idx):
            if distance_mask[idx]:
                return 1.0, 0.0
            val = float(val)
            pdf_idx = min(
                int((val - ming) / (maxg - ming) * len(intens)), len(intens) - 1
            )
            return foreground_pdf[pdf_idx], background_pdf[pdf_idx]
            return foreground_kde(val)[0], background_kde(val)[0]

        def pairwise(val1, val2, idx1, idx2):
            return 3 + 12 * abs(float(val1) - float(val2)) / 255.0

        print("Getting ready to segment")
        estimate = segment_image(grayscale_image, unary, pairwise)

        labeled_array, num_features = label(estimate)

        cleaned_estimate = np.isin(
            labeled_array, np.unique(labeled_array[distance_mask == 1])
        )

        self.set_overlay_mask(cleaned_estimate)

    def initialize_plot(self):
        grayscale_image = self.image.convert("L")
        histogram = np.histogram(grayscale_image, bins=256, range=(0, 256))[0]
        histogram = histogram.astype(float) / histogram.sum()
        self.subplot.clear()
        self.subplot.plot(histogram)
        self.subplot.set_title("Intensity distributions")
        self.plot_canvas.draw()

    def plot_histogram(self):
        grayscale_image = self.image.convert("L")
        histogram = np.histogram(grayscale_image, bins=256, range=(0, 256))[0]
        self.subplot.clear()
        self.subplot.plot(histogram)
        self.subplot.set_title("Grayscale Histogram")
        self.plot_canvas.draw()

    def resize_image(self, image, max_size):
        ratio = min(max_size[0] / image.width, max_size[1] / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        resized_image = image.resize(new_size, Image.ANTIALIAS)
        return resized_image

    def set_overlay_mask(self, mask):
        self.mask = mask
        overlay_image = Image.new("RGBA", self.image.size, (0, 0, 0, 0))
        overlay_array = np.array(overlay_image)

        print(self.mask.shape)
        print(overlay_array.shape)
        overlay_array[:, :, :3] = np.array([0, 0, 255])[None, None, :]
        overlay_array[self.mask, 3] = 120  # alpha channel

        self.overlay = Image.fromarray(overlay_array, "RGBA")
        self.update_canvas()

    def test_overlay(self):
        grayscale = self.image.convert("L")
        image_array = np.array(grayscale)

        threshold = 128
        mask = image_array > threshold
        self.set_overlay_mask(mask)

    def update_canvas(self):
        combined = Image.alpha_composite(self.image.convert("RGBA"), self.overlay)
        self.tk_image = ImageTk.PhotoImage(image=combined)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)


if __name__ == "__main__":
    ImageDrawApp("images/pup.png")
