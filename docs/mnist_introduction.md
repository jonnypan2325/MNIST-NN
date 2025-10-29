{% include navigation.html %}

# Intro to MNIST Dataset

### What is MNIST?

The **MNIST dataset** was created in 1998, specifically for training machine learning models for pattern recognition. The dataset contains **60,000 training images** and **10,000 testing images**.

When training machine learning models, a separate *unseen* testing set is needed for model evaluation. This ensures that the model does not overfit the training set and that the training can be applied generally to unseen data. 

---

### Content of MNIST

The dataset contains grayscale images of handwritten digits along with their corresponding labels:

- **Training Set:** 60,000 labeled images  
- **Test Set:** 10,000 labeled images  
- **Image Size:** 28 × 28  
- **Color Format:** Grayscale (values range from 0 to 255)  
- **Labels:** A single digit from 0 to 9  

---

### Sample MNIST Images

The MNIST dataset contains **handwritten digits from 0 to 9**, collected from a large pool of contributors. Each image is:

- **28×28 pixels**  
- **Grayscale** (pixel values from 0 = black to 255 = white)  
- **Centered and size-normalized** to ensure consistency across digits  

Below is a small sample of images from the dataset. Notice how each digit varies slightly in style and thickness — this variation makes MNIST a good benchmark for testing pattern recognition and machine learning models:

![MNIST Sample Digits](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

- Even though the images are small, they capture enough information for modern algorithms like **neural networks** and **convolutional neural networks (CNNs)** to recognize the digits reliably.  

### Why is MNIST Popular

- Simple and clean: A good starting point for beginners  
- Pre-labeled: Useful for supervised learning  
- Benchmark standard: Many algorithms use MNIST results for comparison  

---

### Use Cases

- Training basic neural networks  
- Experimenting with convolutional neural networks (CNNs)  
- Testing dimensionality reduction techniques (PCA)  

---

### How to Load MNIST with Keras and PyTorch

```python
# ===== KERAS (TensorFlow) =====
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode labels
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

print("Training set:", x_train.shape)
print("Test set:", x_test.shape)


# ===== Pytorch =====
import torch
from torchvision import datasets, transforms

# Define transformations
transform = transforms.Compose([transforms.ToTensor()])

# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")
``` 


---

### Additional Resources
- **Official MNIST Database:** [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)  
- **UCI Machine Learning Repository (MNIST):** [https://archive.ics.uci.edu/dataset/683/mnist+database+of+handwritten+digits](https://archive.ics.uci.edu/dataset/683/mnist+database+of+handwritten+digits)  
- **TensorFlow Datasets (MNIST):** [https://www.tensorflow.org/datasets/catalog/mnist](https://www.tensorflow.org/datasets/catalog/mnist)  
- **PyTorch Datasets (MNIST):** [https://pytorch.org/vision/stable/datasets.html#mnist](https://pytorch.org/vision/stable/datasets.html#mnist)
