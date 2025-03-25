# MNIST Handwritten Digit Classification with TensorFlow

This repository contains a simple deep learning model built using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

## Dataset
The model is trained on the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is a grayscale 28x28 pixel representation of a digit.

## Requirements
Ensure you have the following dependencies installed:
- Python
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

Install dependencies using:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Model Architecture
The model consists of:
- A `Flatten` layer to convert 28x28 images into a 1D array.
- Two `Dense` layers with ReLU activation for feature extraction.
- A final `Dense` layer with Softmax activation for classification into 10 categories.

## Training
The model is compiled with:
- Loss function: `sparse_categorical_crossentropy`
- Optimizer: `Adam`
- Evaluation metric: `accuracy`

Training is performed over 25 epochs with 20% validation split.

## Usage
Run the script to train the model and evaluate its performance:
```bash
python mnist_classification.py
```

## Evaluation
- The model is evaluated on the test dataset using `accuracy_score` from scikit-learn.
- Training and validation loss/accuracy curves are plotted.
- Individual test images can be visualized along with model predictions.

## Results
The trained model makes predictions on test images, and the predicted class is determined using `argmax` on softmax output.

## Visualization
- The script includes code to plot the loss and accuracy curves.
- Test images can be displayed using Matplotlib, along with their predicted labels.

## License
This project is open-source and available under the MIT License.

