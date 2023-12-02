# Project Code Overview

Requirements are in requirements.txt, not guarenteed to be everything

## mrf_cut.py

Code to construct a flow network and solve it to segment a 2d numpy array representing an image

## viztools.py

Code to visualize a networkx graph

## 3_demo.ipynb

Code for section 3, demonstrating a graph cut on a 3x3 pixel

## bar_example.py

Code to generate a bar and add different types of noise to it for section 5

## 5p2_experiment.py

Code to run multiple trials using the bar data and varying beta and degree of noise
Uses multiprocessing, saves the results in a .npy file to be visualized separtely in 5_viz.ipynb

## 5p3_experiment.py

Very similar looking code to 5p2_experiment.py but this time varies the unary potential

## 5_viz.ipynb

Visualizes results from 5p2/3 experiments

## 6p1_video_segment.ipynb

Code to perform video segmentation and visualize it

## 6p2_magic_wand.py

Code for GUI for magic wand using graph cut
