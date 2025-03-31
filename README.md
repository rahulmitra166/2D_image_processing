# 2D Image Processing

This repository contains a Python module for 2D image processing, specifically designed to analyze particles in images. The module provides various functionalities including image loading, normalization, background subtraction, filtering, thresholding, morphological operations, contour detection, and particle property calculations.

## Features

- **Image Loading**: Load individual images or multiple images from a directory.
- **Normalization**: Normalize images using min-max normalization.
- **Background Subtraction**: Calculate and subtract the background using mean, median, or mode.
- **Filtering Techniques**: Apply bilateral filtering, median filtering, and anisotropic diffusion.
- **Contrast Enhancement**: Enhance image contrast using CLAHE.
- **Thresholding**: Apply global (Otsu or manual) and adaptive thresholding.
- **Morphological Operations**: Perform operations such as closing, opening, hole filling, and removal of small objects.
- **Contour Detection**: Detect and filter contours based on area and region-of-interest constraints.
- **Particle Analysis**: Calculate various properties (area, perimeter, equivalent diameter, etc.) for detected particles.

## Requirements

- Python 3.x
- [OpenCV](https://opencv.org/) (`opencv-python`)
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-image](https://scikit-image.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- **feret** and **utils** modules:
  - Ensure that these modules are available (either install via pip or include them in your repository).

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

2. (Optional) Create and activate a virtual environment:

  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate

# Usage

Import the ParticleAnalyzer class and use its methods to process and analyze images. For example:

from 2D_image_processing import ParticleAnalyzer

### Initialize analyzer
analyzer = ParticleAnalyzer()

### Load an image
image = analyzer.load_image('path/to/image.bmp')

### Normalize the image
normalized_image = analyzer.normalize_image(image)

### Enhance contrast using CLAHE
enhanced_image = analyzer.contrast_enhancement(normalized_image)

### Plot pixel histogram of the image
analyzer.plot_pixel_histogram(image)

### Continue with background removal, filtering, thresholding, etc.


# Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.

---

### requirements.txt

```plaintext
opencv-python
numpy
scipy
scikit-image
pandas
matplotlib
# Note: The 'feret' and 'utils' modules must be provided separately or installed as needed.


