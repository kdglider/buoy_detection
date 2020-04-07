# Buoy Detection with Gaussian Mixture Models
This repository is not yet complete

## Overview


## Personnel
Hao Da (Kevin) Dong

Anshuman Singh


## License
This project is licensed under the BSD 3-Clause. Please see LICENSE for additional details and disclaimer.


## Dependencies
The system must have Python 3, NumPy, SciPy and OpenCV installed. Our systems are Windows-based with Python/NumPy installed as part of [Anaconda](https://www.anaconda.com/distribution/), and with OpenCV 4.1.2. 

Begin by installing the pip package management system:
```
sudo wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
```

Use the following commands to install the dependencies after installing Anaconda:
```
pip install numpy
pip install scipy
pip install opencv-python
```
Though untested, using a Linux system and/or using packages of a slightly different version should also work.


## Run Demonstration
To train and save GMM parameters, run:
```
python3 GMM.py
```
The input training set as well as the training parameters can be edited at the bottom of the script. The training set should be a .npy file with an MxD array. Sample sets are provided in the training_sets folder.

To run the buoy detection application on a video using pre-trained GMM parameters, run:
```
python3 BuoyDetector.py
```
The parameter files should be those saved by running GMM.py. These can be modified at the bottom of the script, along with the video file and a flag to save the video.

