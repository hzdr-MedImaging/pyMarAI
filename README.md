# pyMarAI — Tumor Spheroids Auto Delineation Tool

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
  - Optional manual correction hand-off (e.g. to [**ROVER**](http://abx-rover.de/rover/))

- **Continuous dataset growth for retraining**  
  - All new and manually corrected delineations are collected in a centralized directory structure  
  - Facilitates expansion of the training dataset for future retraining and further improvement of the CNN model

## Supported formats
pyMarAI allows to process the following data formats:
- **PNG / TIFF** microscopic images

## Requirements
- **Python 3.10+**
- **nnUNet v2**
- Optional: **ROVER** PET Image Analysis software package

## Build & install
```bash
pip install .
```

## Model installation

TBD

## Citation
If you use pyMarAI (or parts of it) in your own projects, evaluations or publications please cite our work using

```bib
@article{Maus2025,
  title = {Automatic delineation of tumor spheroids in microscopic images using deep-learning}
}
```

## License
pyMarAI is licensed under the **Apache License, Version 2.0.**
See `LICENSE` for the full text.

### Notes on Qt licensing
pyMarAI is released under **Apache-2.0**. It uses **pyQt**, which is
available under **GPLv3**. If you distribute applications that use pyMarAI
you are responsible for **GPL compliance** for pyQt (dynamic linking
recommended, include license texts, do not prohibit relinking, and provide
installation information for locked-down devices).
