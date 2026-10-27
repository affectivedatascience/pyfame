[![Docs](https://github.com/affectivedatascience/PyFAME/actions/workflows/deploy.yml/badge.svg)](https://affectivedatascience.github.io/pyfame/)
[![Tests](https://github.com/affectivedatascience/PyFAME/actions/workflows/tests.yml/badge.svg)](https://github.com/affectivedatascience/PyFAME/actions/workflows/tests.yml)
[![PyPI Version](https://img.shields.io/pypi/v/pyfame)](https://pypi.org/project/pyfame/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pyfame?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/pyfame)

# PyFAME <img src="https://raw.githubusercontent.com/affectivedatascience/PyFAME/master/docs/assets/pyfame_logo.png" width="100" align="right" />

PyFAME - The Python Facial Analysis and Manipulation Environment - is an open-source Python toolbox for manipulating facial features in images and videos. It provides researchers with tools to modify specific facial regions, control how changes develop over time, and combine multiple effects within a single workflow. These tools make it easier to create consistent, reproducible experimental stimuli without requiring extensive programming experience. By simplifying the process of facial image and video manipulation, PyFAME helps researchers explore how different facial features influence human perception.

PyFAME leverages the powerful FaceLandmarker solution from Google MediaPipe, which provides automated detection of 478 unique facial landmarks. By accessing and chaining together landmark indices to form complex paths, many complex image and video manipulations can be performed.

Here is an example of PyFAME performing pixel-level manipulations over video frames to create new ones:

<img src="https://raw.githubusercontent.com/affectivedatascience/PyFAME/master/docs/assets/pyfame_gallery.webp" width=800 align="center"/>

## Statement of Need

Currently, there are no available tools for performing these types of pixel-level operations over videos. Existing research has commonly used general image editing tools (i.e. Photoshop), but these tools often require extensive training and rely on manual editing of video frames.

PyFAME provides users the ability to selectively modify specific regions of the face, in both static images and videos. Our package also seamlessly integrates temporal functions into it's video processing, allowing users to specify how and when pixel-level operations will be applied. These operations can be acheived with as little as 5 lines of code.

Furthermore, the JSON log file outputs are ingestible. This makes it much easier to create consistent, reproducible experimental stimuli without requiring extensie programming experience. By simplifying the process of facial image and video manipulation, PyFAME helps researchers explore how different facial features influence human perception.

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

## Citation

```bibtex
@software{Bosman_PyFAME_the_Python_2026,
    author = {Bosman, Gavin and Livingstone, Steven R.},
    month = oct,
    title = {{PyFAME: the Python Facial Analysis and Manipulation Environment}},
    url = {https://github.com/affectivedatascience/pyfame},
    version = {2.0.0},
    year = {2026}
}
```

## Documentation and Changelog

To view our documentation, examples and tutorials, see [PyFAME Docs](https://affectivedatascience.github.io/pyfame/).

This project maintains a changelog, following the format of [keep a changelog](https://keepachangelog.com/en/1.0.0/). This project also adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## Contributing

Please see the [contributing guide](https://github.com/affectivedatascience/pyfame/tree/master/.github/CONTRIBUTING.md)

## License

[GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)