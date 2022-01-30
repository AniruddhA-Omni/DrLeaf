# HoyaHacks2022


### Important Links
Demonstration: <a href="https://youtu.be/M9AihqaRUgE">Check out</a>  

## Problem Statement

Nowadays, we are witnessing many new viruses and diseases, creating huge damage to human mankind but this problem is not only for human mankind, it is also there in agriculture. Many diseases are affecting plants' lives and it is also affecting the farmers' efforts and their income. They have to spend a lot of time and money to figure out what the disease is and what treatment should be done. There is a need to create a solution to make the process easier and less costly. 

## Our Solution

Here comes our product **DrLeaf** which not only makes the work easier but also reduces the effort and expenditure of the farmer to identify the disease and its treatment methods. We have to upload the image of an affected plant’s leaf through our website and our plant disease prediction model predicts and returns the disease name. And along with the disease name, we also provide the best suitable methods to cure the disease.

## Plant Disease Prediction Model

* We built our prediction model using machine learning and deep learning.
* We have prepared our model using 87000 different images of affected plants. We collected this dataset from Kaggle.
* Our model predicts with an accuracy of **99.27%**.
 
 
## Project Details

* We made a website using HTML, CSS and flask and we deployed our plant disease prediction model in it.
* Users should upload the image of an affected leaf. This image will be sent to our model and our model predicts the disease and shows the output.
* For now, our model predicts the below-mentioned plants. Our plan is to update it to predict most of the diseases. 

  - Apple
  - Cherry
  - Corn
  - Grape
  - Orange
  - Peach
  - Pepper
  - Potato
  - Squash
  - Strawberry
  - Tomato

* In this way, our model predicts the disease and also provides treatment methods.
## <u>Dependencies and Installation</u>

- Python 3.6 and Python 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- Optional: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)


### Installation

1. Clone repo

    ```bash
    git clone https://github.com/AniruddhA-Omni/HoyaHacks2022.git
    cd HoyaHacks2022
    ```
2. Install dependent packages
    ```bash
    pip install -r requirements.txt
   ```

### Run the program
   ```
   python app.py
   ``
