{% include navigation.html %}

# Intro to MNIST Dataset

### What is MNIST?

The MNIST dataset was created in 1998, specifically for training machine learning models for pattern recognition. The dataset contaians 60,000 training images and 10,000 testing images.

When training machine learning models, a separate *unseen* testing set is needed for model evaluation. This ensures that the model does not overfit the training set and the traning can be applied generally to unseen data. 

### Content of MNIST

The dataset contains grayscale images of handwritten digits along with their corresponding labels:

- **Traning Set:** 60,000 labeled images  
- **Train Set:** 10,000 labeled images  
- **Image Size:** 28 X 28  
- **Color Format:** Grayscale (Values range from 0 to 255)  
- **Labels:** A single digit from 0 to 9  

---

### Why is MNIST Popular:

- Simple and Clean: A good tsrting point for begginers  
- Pre-labled: Useful for supervised learning  
- Benchmark standard: Many algorithms use MNIST results for comparision.  

---

### Use Cases

- Traning basis neural networks  
- Experimenting with convolutional neural networks (CNNs)  
- Testing dimensionalty reduction techniques (PCA)  

---