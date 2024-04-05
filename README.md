# Procedural-terrain-generation-with-style-transfer

`abstract`


***In this study we introduce a new technique for the generation of terrain maps, exploiting a combination of procedural generation and Neural Style Transfer. We consider our approach to be a viable alternative to competing generative models, with our technique achieving greater versatility, lower hardware requirements and greater integration in the creative process of designers and developers. Our method involves generating procedural noise maps using either multi-layered smoothed Gaussian noise or the Perlin algorithm. We then employ an enhanced Neural Style transfer technique, drawing style from real world height maps. This fusion of algorithmic generation and neural processing holds the potential to produce terrains that are not only diverse but also closely aligned with the morphological characteristics of real-world landscapes, with our process yielding consistent terrain structures with low computational cost and offering the capability to create customized maps. Numerical evaluations further validate our model’s enhanced ability to accurately replicate terrain morphology, surpassing traditional procedural methods.***

### Project Structure

This repository contains several .py files for both the computation of the procedural noise maps and the transferring of terrain morphology. The project is currently working with tf 2.15 and keras 3, it should be compatible with previous versions as well. My focus is now shifting to deep prior image generators for the creation of the output image, feel free to reach out for the latest implementations, which are currently achieving the best results. 

- `procedural_noise_functions.py`: 
    - **Description**: This file contains all the accessory functions needed to procedurally generate noise

- `generate_noise.py`: 
    - **Description**: A generate file that when called save in the noise foldes a collection of generated noise images in grayscale. Takes as input the dimension of the output image and a flag discriminating between pern and explicit noise 

- `transfer_morphology.py`: 
    - **Description**: Contains the core code for generating transferring of the morphological features. Takes as input the number of necessary iterations. 

### Generating procedural maps

![](https://github.com/fmerizzi/Procedural-terrain-generation-with-style-transfer/blob/main/images/presenting.png)

### Style transferring morphological features

![](https://github.com/fmerizzi/Procedural-terrain-generation-with-style-transfer/blob/main/images/general_results.png)

### Volumetric representation

![](https://github.com/fmerizzi/Procedural-terrain-generation-with-style-transfer/blob/main/images/summary.drawio.png)
