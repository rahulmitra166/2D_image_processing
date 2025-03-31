import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy import stats
from skimage import measure
from skimage.morphology import remove_small_objects
from utils import contour_area, contour_within_mask, bounding_box
import feret
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import json

class ParticleAnalyzer:
    """
    ParticleAnalyzer is a class for processing 2D images to analyze particles.
    It provides methods for image loading, normalization, background subtraction,
    filtering, thresholding, morphological operations, contour detection, and particle property calculation.
    """

    def __init__(self):
        # Initialization method for ParticleAnalyzer.
        pass

    def load_image(self, image_file):
        """
        Load a grayscale image from the specified file.
        :param image_file: Path to the image file.
        :return: Grayscale image.
        """
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        return image

    def all_images_from_path(self, path, ext):
        """
        Retrieve all image files with a specified extension from a given directory and its subdirectories.
        :param path: Path object representing the directory.
        :param ext: File extension to search for (e.g., '*.bmp').
        :return: A tuple containing a list of image file paths and a list of subfolder names.
        """
        all_subfolds = [f.name for f in path.iterdir() if f.is_dir()]
        all_image_files = path.glob("**/" + ext)
        return list(all_image_files), all_subfolds

    def load_images_from_folder(self, folder):
        """
        Load all images from a folder that start with '0' and end with '.bmp'.
        :param folder: Folder path.
        :return: A tuple of (list of images, list of filename prefixes).
        """
        images = []
        filenames = []
        for filename in os.listdir(folder):
            if filename.endswith('.bmp') and filename.startswith('0'):
                img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
                fname = os.path.splitext(filename)[0]
                if img is not None:
                    images.append(img)
                    filenames.append(fname)
        return images, filenames

    def load_json(self, folder):
        """
        Load all JSON files from the specified folder and its subdirectories.
        :param folder: Folder path.
        :return: Dictionary with folder names as keys and JSON content as values.
        """
        json_paths = [os.path.join(root, t) for root, dirs, files in os.walk(folder)
                      if files != [] for t in files if t[-4:] == 'json']
        json_dicts = {}
        for j in json_paths:
            with open(j, 'r', encoding='utf-8', errors='ignore') as f:
                # Use the parent folder name as the key
                name = j.split('\\')[-2]
                json_dicts[name] = json.load(f)
        return json_dicts

    def normalize_all_images(self, images):
        """
        Normalize a list of images using min-max normalization.
        :param images: List of images.
        :return: List of normalized images.
        """
        normalized_images = [
            cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            for image in images
        ]
        return normalized_images

    def normalize_image(self, image):
        """
        Normalize a single image using min-max normalization.
        :param image: Input image.
        :return: Normalized image.
        """
        normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return normalized_image

    def calculate_background_image(self, images, method='mean'):
        """
        Calculate a background image from a list of images using the specified method.
        :param images: List of images.
        :param method: Method for background calculation ('mean', 'median', or 'mode').
        :return: Background image.
        """
        if method == 'mean':
            background_image = np.mean(images, axis=0)
            return background_image
        elif method == 'median':
            background_image = np.median(images, axis=0)
            return background_image
        elif method == 'mode':
            background_image = stats.mode(images, axis=0, keepdims=True).mode[0]
            return background_image
        else:
            raise ValueError(f'Invalid background calculation method: {method}')

    def remove_background_and_clip(self, image, background_image):
        """
        Subtract the background image from the input image and clip the result to the range [0, 255].
        :param image: Input image.
        :param background_image: Background image.
        :return: Image after background removal and clipping.
        """
        image = image.astype(np.float32)
        background_image = background_image.astype(np.float32)
        processed_image = cv2.subtract(image, background_image)
        clipped_image = np.clip(processed_image, 0, 255)
        clipped_image = clipped_image.astype(np.uint8)
        return clipped_image

    def remove_background_local(self, image, kernel_size, sigma=0):
        """
        Remove the local background using Gaussian blur.
        :param image: Input image.
        :param kernel_size: Kernel size for Gaussian blur.
        :param sigma: Gaussian kernel standard deviation.
        :return: Image with local background removed.
        """
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        local_mean = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        background_removed = cv2.subtract(image, local_mean)
        return background_removed

    def plot_pixel_histogram(self, image):
        """
        Plot histograms for both the original and normalized images.
        :param image: Input image.
        """
        norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Create subplots for both histograms
        plt.figure()
        plt.clf()
        plt.gcf().set_size_inches(12, 6)
        plt.subplot(1, 2, 1)
        plt.hist(image.ravel(), 256, [0, 256])
        plt.title('Pixel Histogram of Original Image')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.subplot(1, 2, 2)
        plt.hist(norm_image.ravel(), 256, [0, 256])
        plt.title('Pixel Histogram of Normalized Image')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        plt.close()

    def contrast_enhancement(self, image, method='clahe', clip_limit=3.5, tile_grid_size=12):
        """
        Enhance image contrast using the specified method.
        Currently supports CLAHE (Contrast Limited Adaptive Histogram Equalization).
        :param image: Input grayscale image.
        :param method: Enhancement method ('clahe').
        :param clip_limit: CLAHE clip limit.
        :param tile_grid_size: Grid size for CLAHE.
        :return: Contrast-enhanced image.
        """
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            enhanced_image = clahe.apply(image)
        else:
            raise ValueError(f'Invalid contrast enhancement method: {method}')
        return enhanced_image

    def bilateral_filtering(self, image, min_area=51, sigma_s=2):
        """
        Apply bilateral filtering to the image.
        Note: The parameter 'min_area' is overridden to 1 inside the function.
        :param image: Input image.
        :param min_area: (Overridden) Minimum area for filtering.
        :param sigma_s: Sigma value for spatial domain filtering.
        :return: Bilaterally filtered image.
        """
        min_area = 1  # Overriding parameter to 1
        filtered_image = cv2.bilateralFilter(image, d=min_area, sigmaColor=0, sigmaSpace=0)
        return filtered_image

    def median_filtering(self, image, kernel_size=5):
        """
        Apply median filtering to reduce noise in the image.
        :param image: Input image.
        :param kernel_size: Kernel size for the median filter.
        :return: Median filtered image.
        """
        filtered_image = cv2.medianBlur(image, kernel_size)
        return filtered_image

    def anisodiff(self, image, niter=1, kappa=50, gamma=0.25, step=(1., 1.), option=1):
        """
        Perform anisotropic diffusion to reduce noise while preserving edges.
        :param image: Input image.
        :param niter: Number of iterations.
        :param kappa: Conduction coefficient (edge sensitivity).
        :param gamma: Diffusion speed.
        :param step: Tuple indicating step size in each dimension.
        :param option: Option for the diffusion function (1 or 2).
        :return: Diffused image.
        """
        if image.ndim == 3:
            warnings.warn("Only grayscale images allowed, converting to 2D matrix")
            image = image.mean(2)

        image = image.astype('float32')
        diffused_image = image.copy()

        # Initialize internal variables for diffusion process
        deltaS = np.zeros_like(diffused_image)
        deltaE = deltaS.copy()
        NS = deltaS.copy()
        EW = deltaS.copy()
        gS = np.ones_like(diffused_image)
        gE = gS.copy()

        for ii in range(niter):
            # Calculate differences along South and East directions
            deltaS[:-1, :] = np.diff(diffused_image, axis=0)
            deltaE[:, :-1] = np.diff(diffused_image, axis=1)

            # Compute conduction gradients based on selected option
            if option == 1:
                gS = np.exp(-(deltaS / kappa) ** 2.) / step[0]
                gE = np.exp(-(deltaE / kappa) ** 2.) / step[1]
            elif option == 2:
                gS = 1. / (1. + (deltaS / kappa) ** 2.) / step[0]
                gE = 1. / (1. + (deltaE / kappa) ** 2.) / step[1]

            E = gE * deltaE
            S = gS * deltaS

            # Adjust differences for neighboring pixels
            NS[:] = S
            EW[:] = E
            NS[1:, :] -= S[:-1, :]
            EW[:, 1:] -= E[:, :-1]

            # Update the diffused image
            diffused_image += gamma * (NS + EW)
        diffused_image = diffused_image.astype('uint8')
        return diffused_image

    def threshold_image(self, image, thresh=127, global_method="otsu", adaptive_method="mean", adaptive_block_size=57):
        """
        Apply global and adaptive thresholding to an image.
        Also plots the image histogram for visualization.
        :param image: Input image.
        :param thresh: Threshold value for global thresholding if not using Otsu.
        :param global_method: Global thresholding method ('otsu' or 'normal').
        :param adaptive_method: Adaptive thresholding method ('mean' or 'gaussian').
        :param adaptive_block_size: Block size for adaptive thresholding.
        :return: Binary image after thresholding.
        """
        plt.figure()
        plt.hist(image.ravel(), 256, [0, 256])
        plt.title('Histogram of Input Image')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.close()

        if global_method == 'otsu':
            _, global_binary = cv2.threshold(image, 9, 255, cv2.THRESH_OTSU)
        elif global_method == 'normal':
            _, global_binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
        else:
            raise ValueError(f'Invalid global thresholding method: {global_method}')
        
        if adaptive_method == 'mean':
            adaptive_binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, adaptive_block_size, 0
            )
        elif adaptive_method == 'gaussian':
            adaptive_binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_block_size, 0
            )
        else:
            raise ValueError(f'Invalid adaptive thresholding method: {adaptive_method}')
        
        # Currently returns the global thresholded image; modify as needed to combine with adaptive results.
        combined_binary = global_binary
        return combined_binary

    def morphological_operations(self, binary_image, operation='close', kernel_size=5, iterations=1, min_size=51):
        """
        Apply morphological operations such as hole filling, small object removal,
        and closing/opening to a binary image.
        :param binary_image: Input binary image.
        :param operation: Operation to perform ('close' or 'open').
        :param kernel_size: Size of the structuring element.
        :param iterations: Number of iterations for the morphological operation.
        :param min_size: Minimum object size to retain.
        :return: Morphologically processed image.
        """
        # Fill holes in the binary image
        binary_image = binary_fill_holes(binary_image)
        binary_image = binary_image.astype(np.uint8) * 255
        
        # Remove small objects from the image
        binary_image = remove_small_objects(binary_image, min_size, connectivity=8)
        binary_image = binary_image.astype(np.uint8)
        
        # Create structuring element
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if operation == 'close':
            morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            morph_image = morph_image.astype(np.uint8)
        elif operation == 'open':
            morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
            morph_image = morph_image.astype(np.uint8)
        else:
            morph_image = binary_image
        return morph_image

    def detect_contours(self, binary_image, min_area, max_area):
        """
        Detect contours in a binary image and filter them based on area and location constraints.
        :param binary_image: Binary image.
        :param min_area: Minimum acceptable contour area.
        :param max_area: Maximum acceptable contour area.
        :return: List of filtered contours.
        """
        # Get contours using skimage
        contours = measure.find_contours(binary_image, level=1)

        # Filter contours by area and ensure they fall within a defined region of interest
        filtered_contours = [
            contour for contour in contours
            if min_area <= contour_area(contour) <= max_area and
               contour_within_mask(contour, binary_image.shape, 1250)
        ]
        return filtered_contours

    def calculate_particle_properties(self, binary_image, contours, experiment, image, min_area, conversion_factor):
        """
        Calculate properties (such as area, perimeter, equivalent diameter, etc.) for each detected particle.
        The properties are returned as a pandas DataFrame.
        :param binary_image: Binary image used for detection.
        :param contours: List of detected contours.
        :param experiment: Experiment identifier.
        :param image: Image identifier.
        :param min_area: Minimum area threshold.
        :param conversion_factor: Factor to convert pixel measurements to real-world units.
        :return: DataFrame containing particle properties.
        """
        properties = pd.DataFrame(columns=[
            'experiment', 'image', 'label', 'area', 'perimeter',
            'equivalent_diameter', 'max_feret', 'min_feret',
            'axis_major', 'axis_minor', 'solidity', 'aspect_ratio', 
            'circularity', 'convexity', 'eccentricity'
        ])
        print(f'Number of contours detected: {len(contours)}')
        
        # Compute bounding boxes and extract regions for each contour
        boxes = [bounding_box(cnt) for cnt in contours]
        cnt_images = [binary_image[x:x + w, y:y + h] for x, y, w, h in boxes]
        region_properties = [measure.regionprops(measure.label(img)) for img in cnt_images]

        # Populate DataFrame with calculated properties
        object_data_temp = pd.DataFrame()
        object_data_temp['experiment'] = [experiment] * len(contours)
        object_data_temp['image'] = [image] * len(contours)
        object_data_temp['label'] = [i for i in range(len(contours))]
        object_data_temp['area'] = [rp[0].area * conversion_factor ** 2 for rp in region_properties]
        object_data_temp['perimeter'] = [rp[0].perimeter * conversion_factor for rp in region_properties]
        object_data_temp['equivalent_diameter'] = np.sqrt(4 * object_data_temp['area'] / np.pi)
        object_data_temp['max_feret'] = [feret.max(img) * conversion_factor for img in cnt_images]
        object_data_temp['min_feret'] = [feret.min(img) * conversion_factor for img in cnt_images]
        object_data_temp['axis_major'] = [rp[0].major_axis_length * conversion_factor for rp in region_properties]
        object_data_temp['axis_minor'] = [rp[0].minor_axis_length * conversion_factor for rp in region_properties]
        object_data_temp['solidity'] = [rp[0].solidity for rp in region_properties]
        object_data_temp['aspect_ratio'] = object_data_temp['min_feret'] / object_data_temp['max_feret']
        object_data_temp['circularity'] = 4 * object_data_temp['area'] / (np.pi * object_data_temp['max_feret'] ** 2)
        object_data_temp['convexity'] = [max(rp[0].convex_area / rp[0].area, 1) for rp in region_properties]
        object_data_temp['eccentricity'] = [rp[0].eccentricity for rp in region_properties]

        if len(properties) > 0:
            properties = pd.concat([properties, object_data_temp])
        else:
            properties = object_data_temp
        return properties if not properties.empty else None