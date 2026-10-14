[![Docs](https://github.com/affectivedatascience/PyFAME/actions/workflows/deploy.yml/badge.svg)](https://affectivedatascience.github.io/pyfame/)
[![Tests](https://github.com/affectivedatascience/PyFAME/actions/workflows/tests.yml/badge.svg)](https://github.com/affectivedatascience/PyFAME/actions/workflows/tests.yml)
[![PyPI Version](https://img.shields.io/pypi/v/pyfame)](https://pypi.org/project/pyfame/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pyfame?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/pyfame)

# PyFAME: the Python Facial Analysis and Manipulation Environment
<img src="https://raw.githubusercontent.com/affectivedatascience/PyFAME/master/docs/assets/pyfame_logo.png" width=300 />

PyFAME is a python package for dynamic region occlusion and skin colour modification of faces in video and images. Provides a range of tools for changing the colour of user-specified facial regions using industry-standard colour spaces (L*a*b, HSV, BGR), occlusion of user-defined facial regions (e.g., eyes, nose, mouth, hemi-face), and isolation of the head from background scene through video matting. Facial modifications can be further transitioned on and off through a range of timing functions (e.g., linear, sigmoid, Gaussian etc).

PyFAME leverages the powerful Face Mesh solution from MediaPipe, which provides automated detection of 478 unique facial landmarks. By accessing the x-y pixel coordinates of these landmarks, many complex image and video manipulations can be performed.

Here is an example of PyFAME performing pixel-level modifications to create new videos from the original one:

<img src="https://raw.githubusercontent.com/affectivedatascience/PyFAME/master/docs/assets/grid_display.png" width=400/>

## Statement of Need

Manipulation of facial images is a common paradigm in psychology and related disciplines. Many studies use stand-alone GUI-based editors (e.g., photoshop). However, these tools lack dedicated facial manipulation types, require significant domain expertise, work primarily with images, lack a reproducible log of changes, and require point-and-click creation of individual images.

PyFAME - the Python Facial Analysis and Manipulation Environment - fills this gap by providing a flexible pipeline for applying and stacking manipulations to facial images, with a scalable design suited for large image or movie datasets.

## Example: 
As a quick example, this is the simplest way to set up your file system and to get processing facial manipulations with PyFAME:
```python
import pyfame as pf

# Create the necessary file system structure to load and organize data
pf.make_paths()

# Get dataframe of file paths, pre-loaded with sample data
files = pf.load_sample_data()

# Create your manipulation layer
layer_recolour = pf.layer_colour_recolour()

# Apply your layer to your file paths
pf.apply_layers(file_paths = files, layers = layer_recolour)
```

## Documentation and Changelog

To view our documentation, examples and tutorials, see [PyFAME Docs](https://affectivedatascience.github.io/pyfame/).

This project maintains a changelog, following the format of [keep a changelog](https://keepachangelog.com/en/1.0.0/). This project also adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## Contributing

Please see the [contributing guide](https://github.com/affectivedatascience/pyfame/tree/master/.github/CONTRIBUTING.md)

## License

[GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)