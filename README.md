# mnist-draw
This repository contains a single page website that enables users to hand-draw and classify clothes using machine learning. A machine learning model trained against the MNIST fashion dataset is used for classification. 

the repository forked from https://github.com/rhammell/mnist-draw which works on mnist digits dataset, I changed the code and the trained model to work with mnist fashion dataset.
I used this colab link https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l03c01_classifying_images_of_clothing.ipynb?fbclid=IwAR2diCMdq46j6O7w4OI-_DDTevgatE0X3qpJ7Ut7-ww-_GKmj_34GgfDbYY
to train the model.

# Setup 
Python 3.5+ is required for compatability with all required modules

```bash
# Clone this repository
git clone https://github.com/rhammell/mnist-draw.git

# Go into the repository
cd mnist-draw

# Install required modules
pip install -r requirements.txt
```

# Usage
To launch the website, begin by starting a Python server from the repository folder:
```bash
# Start Python server
python -m http.server --cgi 8000
```
Then open a browser and navigate to `http://localhost:8000/index.html` to view it. 

An example of the website's interface is shown below. Users are guided to draw a digit (0-9) on the empty canvas and then hit the 'Predict' button to process their drawing. Allow up to 1 minute for the processing to complete. Any errors during processing will be indicated with a warning icon and printed to the console. 

Results are displayed as a bar graph where each classification label recieves a score between 0.0 and 1.0 from the machine learning model. Clear the canvas with the 'Clear' button to draw and process other digits.  


## Machine Learning Model
Python scripts related to defining, training, and implementing the machine learning model are contained within the `cgi-bin` folder. 

A convolutional neural network (CNN) is defined within the `model.py` module using the [TFLearn](http://tflearn.org/) library. This model is configured for MNIST data inputs. 

The defined CNN can be trained against the MNIST dataset by running the `train.py` script. This script will automaticallly load the MNIST dataset from the TFLearn library to use as input, and the trained model's parameter files are saved into the `models` directory. Pre-trained model files are made available in this directory already.

The `mnist.py` script implements this trained model against the user's hand-drawn input. When the 'Predict' button is clicked, the contents of the drawing canvas are posted to this script as data url, and a JSON object containing the model's predictions is returned. 

