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
from mrf_cut3 import segment_image_gco
from mrf_cut5 import segment_image_igraph
from mrf_cut2 import segment_image_maxflow


def create_distance_array(image_shape, draw_history):
    """
    Create a 2D array with the same shape as the image, where each element
    is the minimum distance from that point to any point in the draw history.

    Parameters:
    - image_shape: A tuple (height, width) representing the image dimensions.
    - draw_history: A list of tuples, where each tuple is (x, y) coordinates.

    Returns:
    - A 2D numpy array representing the minimum distances.
    """
    # Generate all points in the image
    # all_points = np.indices(image_shape).reshape(2, -1).T
    y_indices, x_indices = np.indices(image_shape)
    all_points = np.stack((x_indices, y_indices), axis=-1).reshape(-1, 2)

    # Convert draw_history to a NumPy array
    draw_history_array = np.array(draw_history)

    # Compute distances from all points to the draw_history points
    distances = cdist(all_points, draw_history_array)

    # Find the minimum distance to draw_history for each point
    min_distances = distances.min(axis=1)

    # Reshape the array back to the image shape
    distance_array = min_distances.reshape(image_shape)

    return distance_array


class ImageDrawApp:
    def __init__(
        self,
        image_path,
        primary_color="red",
        secondary_color="green",
        max_size=(400, 400),
    ):
        # Load and resize the image
        self.original_image = Image.open(image_path)
        self.image = self.resize_image(self.original_image, max_size)
        self.image = self.image.convert("L").convert("RGBA")
        self.image_shape = self.image.size[::-1]
        self.current_color = primary_color
        self.primary_color = primary_color
        self.mesh_color = (0, 0, 255, 120)  # Semi-transparent blue (R, G, B, A)

        self.secondary_color = secondary_color
        self.overlay = Image.new("RGBA", self.image.size)
        self.colors = [primary_color, secondary_color]
        self.draw_mode = 0
        self.eyedropper_mode = False

        # Initialize drawing history
        self.draw_history = []

        # Initialize the root GUI window
        self.root = tk.Tk()
        self.root.title("Draw on Image App")

        # Convert the Image object to a TkPhoto object
        self.tk_image = ImageTk.PhotoImage(self.image)

        # Create a canvas with the size of the resized image
        self.canvas = tk.Canvas(
            self.root, width=self.tk_image.width(), height=self.tk_image.height()
        )
        # self.canvas.pack()
        self.canvas.pack(side="left", fill="both", expand="yes")

        # Add the image to the canvas
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        # Setup the ImageDraw object
        self.draw = ImageDraw.Draw(self.image)
        # self.overlay_draw = ImageDraw.Draw(self.overlay)

        self.draw_distance = np.ones(self.image_shape, int)
        self.mask = np.zeros(self.image_shape, int)
        self.set_overlay_mask(self.mask)
        # self.test_overlay()

        # Bind mouse events to the canvas
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", lambda *args: self.update_canvas())

        # Set up the matplotlib plot
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.subplot = self.figure.add_subplot(111)

        # Canvas for matplotlib plot
        self.plot_canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.plot_widget = self.plot_canvas.get_tk_widget()
        self.plot_widget.pack(side="right", fill="both", expand=True)
        # Plot an initial plot, maybe a histogram of the grayscale image
        self.initialize_plot()

        # Add a button to change colors
        self.color_button = tk.Button(
            self.root, text="Change Color", command=self.toggle_color
        )
        self.color_button.pack()

        # Add a button to change colors
        self.plot_button = tk.Button(
            self.root, text="Update Plot", command=self.update_plot
        )
        self.plot_button.pack()

        # Eyedropper functionality setup
        self.info_label = tk.Label(self.root, text="Color info will be shown here")
        self.info_label.pack(side="bottom")

        # Toggle button for eyedropper mode
        self.eyedropper_button = tk.Button(
            self.root, text="Eyedropper", command=self.toggle_eyedropper
        )
        self.eyedropper_button.pack(side="bottom")

        self.canvas.bind("<Motion>", self.on_canvas_hover)

        # Start the GUI
        self.root.mainloop()

    def toggle_eyedropper(self):
        self.eyedropper_mode = not self.eyedropper_mode

    def on_canvas_hover(self, event):
        if self.eyedropper_mode:
            # Get the color and intensity of the pixel under the cursor
            x, y = event.x, event.y
            if x < self.image.width and y < self.image.height:
                pixel_value = self.image.getpixel((x, y))
                self.info_label.config(text=f"Color: {pixel_value}")
                self.update_histogram_marker(pixel_value[0])  # Assuming grayscale

    def update_histogram_marker(self, intensity):
        # Update the matplotlib plot with a vertical line at the given intensity
        self.subplot.clear()
        self.subplot.axvline(x=intensity, color="r", linestyle="--")
        self.draw_plot(False)

    def paint(self, event):
        # Record the position where the user is drawing
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
    #     # Update the TkPhoto object when the user releases the mouse button
    #     self.tk_image = ImageTk.PhotoImage(self.image)
    #     self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def toggle_color(self):
        # Toggle the current drawing color
        self.draw_mode = (self.draw_mode + 1) % 2
        self.current_color = self.colors[self.draw_mode]

        # Update the drawing with the new color
        # for x1, y1, x2, y2 in self.draw_history:
        #     self.draw.ellipse(
        #         [x1, y1, x2, y2], fill=self.current_color, outline=self.current_color
        #     )

        # Update the TkPhoto object to reflect the color changes
        # self.tk_image = ImageTk.PhotoImage(self.image)
        # self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

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

    def draw_plot(self, clear=True):
        if clear:
            self.subplot.clear()
        self.subplot.plot(self.intens, self.foreground_pdf)
        self.subplot.plot(self.intens, self.background_pdf)
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
        ming = grayscale_image.min()
        maxg = grayscale_image.max()
        self.intens = np.linspace(0, 255, 100)
        ming = self.intens.min()
        maxg = self.intens.max()
        self.foreground_pdf = foreground_kde(self.intens)
        self.background_pdf = background_kde(self.intens)
        # self.subplot.plot(background_kde(intens))

        def unary(val, idx):
            if distance_mask[idx]:
                return 1.0, 0.0
            val = float(val)
            pdf_idx = min(
                int((val - ming) / (maxg - ming) * len(self.intens)),
                len(self.intens) - 1,
            )
            return self.foreground_pdf[pdf_idx], self.background_pdf[pdf_idx]
            return self.background_pdf[pdf_idx], self.foreground_pdf[pdf_idx]
            return foreground_kde(val)[0], background_kde(val)[0]

        def unary(image):
            flat_image = image.ravel()
            flat_distance_mask = distance_mask.ravel()

            # Initialize the unary potentials with the default values
            unary_potentials = np.zeros((flat_image.size, 2), dtype=np.float32)

            # Where the distance mask is true, set the unary potential to (1.0, 0.0)
            unary_potentials[flat_distance_mask, 0] = 1.0
            unary_potentials[flat_distance_mask, 1] = 0.0

            # Calculate the indices into the PDF arrays for the rest of the pixels
            pdf_indices = np.clip(
                ((flat_image - ming) / (maxg - ming) * len(self.foreground_pdf)).astype(
                    int
                ),
                0,
                len(self.foreground_pdf) - 1,
            )

            # Set the unary potentials using the PDF values where the distance mask is false
            unary_potentials[~flat_distance_mask, 0] = self.foreground_pdf[
                pdf_indices[~flat_distance_mask]
            ]
            unary_potentials[~flat_distance_mask, 1] = self.background_pdf[
                pdf_indices[~flat_distance_mask]
            ]

            return unary_potentials.T

        def pairwise(val1, val2, idx1, idx2):
            # return 2 + 12 * abs(float(val1) - float(val2)) / 255.0
            p_agree = 1 - abs(float(val1) - float(val2)) / 255.0
            assert p_agree <= 1 and p_agree >= 0
            return 2 + 3 * (p_agree)
            if p_agree < 0.5:
                # then I'd technically want to penalize for them agreeing, but I'll just not reward them
                return 0
            return 5  # now I want to reward them?

            assert p > 0
            return p

        print("Getting ready to segment")
        estimate = segment_image_igraph(grayscale_image, unary, pairwise)
        # Label the connected components
        labeled_array, num_features = label(estimate)

        self.draw_plot()

        cleaned_estimate = np.isin(
            labeled_array, np.unique(labeled_array[distance_mask == 1])
        )

        # # Find the size of each connected component
        # sizes = ndi_sum(estimate, labeled_array, range(num_features + 1))

        # # Find the label of the largest connected component (excluding the background)
        # largest_label = (
        #     sizes[1:].argmax() + 1
        # )  # The +1 is because we ignore the background label which is 0.

        # # Create a new mask where we keep only the largest connected component
        # cleaned_estimate = labeled_array == largest_label
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
        # Calculate histogram
        grayscale_image = self.image.convert("L")
        histogram = np.histogram(grayscale_image, bins=256, range=(0, 256))[0]
        self.subplot.clear()
        self.subplot.plot(histogram)
        self.subplot.set_title("Grayscale Histogram")
        self.plot_canvas.draw()

    def resize_image(self, image, max_size):
        """
        Resizes the image to fit within a max width and height while maintaining aspect ratio.
        """
        ratio = min(max_size[0] / image.width, max_size[1] / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        resized_image = image.resize(new_size, Image.ANTIALIAS)
        return resized_image

    def set_overlay_mask(self, mask):
        self.mask = mask
        # Create an RGBA image for overlay from the mask
        overlay_image = Image.new("RGBA", self.image.size, (0, 0, 0, 0))
        overlay_array = np.array(overlay_image)

        print(self.mask.shape)
        print(overlay_array.shape)
        # Use the mask to set the alpha channel to 120 where the mask is True
        overlay_array[:, :, :3] = np.array([0, 0, 255])[
            None, None, :
        ]  # Set color to blue
        overlay_array[self.mask, 3] = 120  # Set alpha channel

        # Convert the NumPy array back to a PIL Image and save it as the overlay
        self.overlay = Image.fromarray(overlay_array, "RGBA")
        # Update the overlay on the canvas
        self.update_canvas()

    def test_overlay(self):
        # Convert the image to grayscale
        grayscale = self.image.convert("L")

        # Create a NumPy array from the grayscale image
        image_array = np.array(grayscale)

        # Create a boolean mask where the intensity is greater than a threshold (0.5 in this case)
        threshold = (
            128  # Since pixel values range from 0 to 255, 0.5 corresponds to 128
        )
        mask = image_array > threshold
        self.set_overlay_mask(mask)

    def update_canvas(self):
        # Merge the drawing with the overlay and then with the image
        combined = Image.alpha_composite(self.image.convert("RGBA"), self.overlay)
        self.tk_image = ImageTk.PhotoImage(image=combined)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)


# Create an instance of the app
if __name__ == "__main__":
    ImageDrawApp("images/pup.png")
