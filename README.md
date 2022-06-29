# Introduction

This repository implements the algorithm in "Multi-task learning for predicting SARS-CoV-2 antibody escape" by Barak Gross and Prof. Roded Sharan.
In the paper, the authors aim to prove that using multi-task learning on antibody recognition is a useful paridigm that should get more attention.
For any problems or questions regarding the code, please email barak_gross@mail.tau.ac.il.

# Strcture
The repository contains several folders:
* data - contains two folders: modified and original. Original is the csv we pulled from other sources such as other studies, while modified contains aggregation of them in order to be able to read them more comfortably.
* code - contains the needed code to run to produce the outputs

# How to Run

First make sure the following are installed:
* biopyton
* keras
* tensorflow
* pandas
* numpy
* sklearn

python main.py should create the various models for continuous multi task scenarios. Those models can be passed later on in order to explore and visualize the effects of teh embeddings. This script also create teh various ".pkl" file that contains the information to be unpacked from the data.
How to read the ".pkl" and use the embeddings can be seen in the jupyter notebook.

