# ArrowDetectorAUV
ArrowDetector is a Python tool using OpenCV to detect arrows in images and calculate their orientation.
It leverages color masking, contour detection, and line fitting to determine the direction of the arrow.

Features
Red Masking: Isolates red-colored arrows using HSV color space.
Edge Detection: Uses Gaussian blur and Canny edge detection to enhance contours.
Contour Analysis: Identifies the largest contour, assumed to be the arrow.
Orientation Calculation: Fits a line to the arrow's contour and calculates the angle relative to the vertical axis.
