# AuthenticityAI

## Overview
How can you make sure that a selfie video - captured by a phone - is really showing the person who created that specific recording? The project goal is to build a solution, which is able to tell from a short video, that the person on the video is real, the recording was captured by the phone itself and optionally validate the audio on the recording as well. 

## Features & Tech stack
* **Computer vision and image processing** - OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It provides various tools and algorithms to perform tasks like image and video analysis, object detection, and machine learning in real-time.
* **Face and facial elements recognition** - Haar Cascade is a machine learning-based approach used for object detection in images. It uses features called Haar-like features and employs a cascade function to identify objects. This method is particularly efficient for detecting objects in images, making it a cornerstone for various applications involving face detection, pedestrian detection, and more.
* **Blur detection** - Relies on the variance of Laplacian, a technique adept at identifying partial blurring within a video stream. By calculating the variance of the image gradients, this method effectively discerns regions where blurring may compromise image clarity and may indicate deepfakes. 
* **Face element distortion** - Incorporates a method for detecting potential face element distortion by meticulously examining the coordinates of the eyes and the center of the head. Through this process, the system calculates the internal angles of a triangle formed by these facial landmarks. By ensuring a consistent proportion between the eyes and the face throughout the video stream, the system leverages this unique identifier—reflecting the typical eye-to-face proportion in a human visage—to identify potential distortions in facial elements. 
* **Color histogram analysis** - Analyzing the artifact bounding box contents (eyes, mouth, face, etc.) by color histogram. Comparing histograms with median and variance, deepfake videos have higher probability of larger output values. 

## Working principle
* Each feature listed above have it's own confidence score. The median and variance of the confidence scores will result the overall performance of the input. 
* Each of the feature analysis can be fine tuned and the limits can be set. 

## Advantages
* Does not rely on third party online services, you can run offline on video files batched. 
* Processing does not require heavy equipment, it can rely CPU only in a server backend environment.
* Does not use too many point meshes, bounding box calculations predetermined to the task performance.

## Plans
* Speech to Text recognition and timestamping with sttcast and vosk - comparing it to mouth artifact (bounding box) movement/variance
* Define multiple triangles across the face to use multiple point sources
* Ear detection and comparison to the triangle values (head is turned sideways?)

## Install & Usage
1. Install the necessary packages from requirements.txt
2. Decide wether you want to use webcam input or not. If not, input file is necessary.
2. ./main.py [--webcam:True|False] [input_file]
3. Results will be displayed on standard output. You may redirect it to an output file.
4. Parsing the results:
	1. Blurred rate: gives an output if the video is blurred. **Range: 0-1, 3 decimals**. Higher numbers may indicate deepfakes. Normally it should stay below 0.02
	2. Variance median: gives an output of the face element processing. **Range: 0-infinite, 3 decimals**. The higher the number, the probability of deepfake is higher. Normally it should stay below 30. 
	3. Success rate: **Range: 0-1, 3 decimals**, gives an information about how many frames were able to detect a fully fledged face. Depends on video quality, it should stay above 0.4.

## Team
* Martin Polyak - ([polyakmartin@draken.hu](mailto:polyakmartin@draken.hu))
* Krisztian Gava - ([gavakrisztian@draken.hu](mailto:gavakrisztian@draken.hu))
* Szabolcs Viktor Ladik - ([ladikszabolcs@draken.hu](mailto:ladikszabolcs@draken.hu))  

[Draken OÜ](https://draken.ee)