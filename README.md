# Main Code of 3D-sEMG-SOM

## Introduction

This repository contains the codebase for the paper "Hand Gesture Recognition with Switching State Based on Surface Electromyography through 3D Self-Organizing Mapping Network." The project focuses on hand gesture recognition using surface electromyography (sEMG) data and employs a 3D-sEMG-SOM (3D Self-Organizing Map) for classification. The project also includes two comparative methods: Long Short-Term Memory (LSTM) networks and Wavelet Neural Networks (WNN).

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [File Structure](#file-structure)
- [Contact Us](#contact-us)
- [Data Usage Statement](#data-usage-statement)

## Features

- Implements the 3D-sEMG-SOM algorithm for gesture classification.
- Compares the classification performance with LSTM and WNN algorithms.
- Includes data preprocessing, feature extraction, model training, and result visualization.

## Requirements

- Python 3.10.9
- Dependencies: `torch`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `tqdm`

## File Structure

```bash
MainCode/
│
├── README.md                    # Project description file
├── 3D_sEMG_SOM.py               # Implementation of the 3D-sEMG-SOM algorithm
├── Compare_LSTM.py              # Implementation of the LSTM algorithm
├── Compare_WNN.py               # Implementation of the WNN algorithm
├── feature_extraction.py        # Feature extraction module
└── util_functions.py            # Utility functions module
```

## Contact Us

- Email: duanyh@mial.nankai.edu.cn
- Email: 2330126751@qq.com

## Data Usage Statement

Due to privacy and ethical considerations, we cannot share the sEMG data used in this project. The data involves participants' physiological signals, which may contain sensitive information. Without explicit consent from the participants, the data cannot be shared or published. If you require similar data for research or development, please contact the appropriate data providers to ensure proper authorization and ethical approval.

If you have any questions regarding data handling or require further explanations, please feel free to contact us.