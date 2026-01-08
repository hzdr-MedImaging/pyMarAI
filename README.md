# pyMarAI: nnU-Net-based Tumor Spheroids Auto Delineation
[![DOI](https://zenodo.org/badge/doi/10.14278/rodare.4198.svg)](https://doi.org/10.14278/rodare.4198)
[![License](https://img.shields.io/github/license/hzdr-MedImaging/pyMarAI.svg)](https://github.com/hzdr-MedImaging/pyMarAI/blob/master/LICENSE)

**pyMarAI** is a toolchain including a PyQt5-based graphical user interface that allows to apply a CNN-based delineation workflow for bioimaging-driven tumor spheroid growth assays commonly used in cancer research.

The software provides a complete pipeline for handling microscopic spheroid image data, running deep-learning–based delineation, and curating results for continuous model improvement.

## Features

- **User-friendly GUI (PyQt5)**  
  - Intuitive interface for configuring and running CNN-based delineation workflows without having to work directly on the command line.

- **Image data management**  
  - Import and organize spheroid image datasets  
  - Automatic conversion between supported image formats (PNG, TIFF)
  - Management of large-scale spheroid experiments

- **CNN-based delineation at scale**  
  - Run automated delineation on large spheroid datasets  
  - Support for GPU-accelerated environments to speed up predictions

- **Quality review and curation**  
  - Visual review of delineated images  
  - Tagging of delineation quality
  - Optional manual correction hand-off to external applications

- **Continuous dataset growth for retraining**  
  - All new and manually corrected delineations are collected in a centralized directory structure  
  - Facilitates expansion of the training dataset for future retraining and further improvement of the CNN model

## Supported formats
pyMarAI allows to process the following data formats:
- **PNG / TIFF** microscopic images

## Requirements
- **Python 3.10+**
- **nnUNet v2**

## Build & install
```bash
pip install .
```

## Model installation
[![DOI](https://zenodo.org/badge/doi/10.14278/rodare.4198.svg)](https://doi.org/10.14278/rodare.4198)
[![License](https://img.shields.io/github/license/hzdr-MedImaging/pyMarAI.svg)](https://github.com/hzdr-MedImaging/pyMarAI/blob/master/MODEL_LICENSE)

TBD

## Citation
If you use pyMarAI (or parts of it) in your own projects, evaluations or publications please cite our work using

```bib
@article{Maus2025,
  title = {Automatic delineation of tumor spheroids in microscopic images using deep-learning}
}
```

## Disclaimer (Research Use Only – Not a Medical Device)

This software and any bundled or referenced model weights are provided **exclusively for research and development
purposes**. They are **not intended** for use in the diagnosis, cure, mitigation, treatment, or prevention of disease,
or for any other clinical decision-making.

- The software is **not** a medical device and is **not** CE-marked.
- No clinical performance, safety, or effectiveness is claimed or implied.
- Any results must not be used to guide patient management.
- Users are responsible for compliance with all applicable laws, regulations, and data protection requirements when
  processing data.

THE SOFTWARE AND MODELS ARE PROVIDED “AS IS”, WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED.

## Licenses

The **code** in this repository is licensed under **Apache-2.0** (see [`LICENSE`](LICENSE)).  
The **model weights** are licensed under **CC-BY-SA-4.0** (see [`MODEL_LICENSE.md`](MODEL_LICENSE.md)).

## Third-Party Licenses

This project uses or interoperates with the following third-party components:

- **nnU-Net v2** – Copyright © respective authors.
    - License: **Apache-2.0**
- **PyTorch**, **NumPy**, **Nibabel**, etc.
    - Licensed under their respective open-source licenses.

Each third-party component is the property of its respective owners and is provided under its own license terms. Copies
of these licenses are available from the upstream projects.

### Notes on Qt licensing
pyMarAI itself is released under **Apache-2.0**. It uses **pyQt**, which is
available under **GPLv3**. If you distribute applications that use pyMarAI
you are responsible for **GPL compliance** for pyQt (dynamic linking
recommended, include license texts, do not prohibit relinking, and provide
installation information for locked-down devices).
