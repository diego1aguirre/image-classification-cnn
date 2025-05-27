# Enhancing Image Classification Using a Convolutional Neural Network Model on the CIFAR-10 Dataset

## Authors
Diego Aguirre

## Project Overview
This project implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The implementation focuses on improving classification performance through data augmentation, dropout regularization, and adaptive learning rate optimization using the Adam optimizer.

## Dataset
The CIFAR-10 dataset consists of:
- 60,000 32x32 color images
- 10 different classes (airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks)
- 50,000 training images and 10,000 test images
- Each class has 6,000 images (5,000 for training, 1,000 for testing)

## Project Structure

## Requirements
The project uses the following key dependencies:
```python
tensorflow>=2.x
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
```

You can install all required packages using:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python
```

## Model Architecture
The CNN model is implemented using TensorFlow/Keras with the following architecture:
- Input Layer: 32x32x3 (RGB images)
- Convolutional Layers with Batch Normalization
- MaxPooling Layers
- Dropout Layers for regularization
- Dense Layers for classification
- Output Layer: 10 units (one for each class)

The model uses:
- ReLU activation functions
- Adam optimizer
- Categorical cross-entropy loss
- Data augmentation for improved generalization
- Early stopping to prevent overfitting

## Training Results
The model was trained for multiple epochs with the following performance metrics:
- Final Training Accuracy: ~85.86%
- Final Validation Accuracy: ~85.18%
- Best Validation Accuracy: ~86.81%

The training process shows:
- Steady improvement in both training and validation accuracy
- Good generalization with minimal overfitting
- Effective use of regularization techniques

## Usage
1. Clone this repository
2. Install the required dependencies
3. Open the Jupyter notebook:
```bash
jupyter image-classification-cnn.ipynb
```
4. Run the cells in sequence to:
   - Load and preprocess the CIFAR-10 dataset
   - Define and train the CNN model
   - Evaluate the model's performance
   - Visualize results and predictions

## Key Features
- Data preprocessing and normalization
- Data augmentation for improved generalization
- Batch normalization for better training stability
- Dropout layers for regularization
- Early stopping to prevent overfitting
- Comprehensive visualization of training progress
- Model evaluation with confusion matrix and classification report

## Future Improvements
Potential areas for enhancement:
1. Experiment with different CNN architectures (e.g., ResNet, DenseNet)
2. Implement learning rate scheduling
3. Try different data augmentation techniques
4. Add model ensemble methods
5. Implement cross-validation
6. Add model interpretability tools
7. Experiment with transfer learning using pre-trained models

## References
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
- [Deep Learning for Computer Vision](https://www.deeplearningbook.org/contents/convnets.html)


