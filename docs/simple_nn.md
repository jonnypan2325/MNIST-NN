{% include navigation.html %}

# A simple Neural Network

# **How Our AI Recognizes Digits: A Math-Based Guide**

This project builds a 2-layer neural network from scratch using only numpy, a powerful math library for Python.

The goal isn't just to *get* the right answer, but to understand the *exact* mathematical process our model uses to "learn." We are building the engine that libraries like TensorFlow and Keras hide from us.

### **The Data: Preparing Images for Math**

Before our model can learn, we must prepare its "textbooks"—the MNIST dataset. The data starts as a collection of 28x28 pixel grids, but our math functions need the data in a very specific format.

* Flattening an Image:  
  A neural network doesn't "see" a 2D image. It sees a list of numbers. "Flattening" is the process of taking a 2D, 28x28 pixel grid and "unrolling" it into a single, 1D list (or "vector") of 784 numbers (since $28 \\times 28 \= 784$).  
  * **Why?** Our model's core math is matrix multiplication. This operation requires a 1D vector as an input, not a 2D grid.  
* Normalization:  
  Each pixel has a value from 0 (black) to 255 (white). We "normalize" this by dividing every pixel's value by 255\.  
  * **Why?** This scales all our input data to be between 0 and 1\. Working with small, consistent numbers helps the learning process (gradient descent) work much faster and more stably.  
* Transposing (Flipping the Data):  
  After loading, our data is shaped as (num\_images, 784). We use .T (transpose) to flip it to (784, num\_images).  
  * **Why?** This is a technical choice for how our math is written. Our model will process all images in a "batch" at once. By setting it up this way, our weight matrix W can multiply the entire dataset X with a single W.dot(X) operation, which is extremely fast.

### **The Model's "Brain": A 2-Layer Network**

Our network is simple, with three main parts:

1. **Input Layer (784 Neurons):** One neuron for every pixel in the flattened image.  
2. **Hidden Layer (10 Neurons):** An "intermediate" layer. You can think of this as the "thinking" part of the brain. It finds patterns in the pixels.  
3. **Output Layer (10 Neurons):** One neuron for each possible digit (0, 1, 2, 3, 4, 5, 6, 7, 8, 9).

This "brain" is just a collection of numbers called **Parameters**, which are stored in the code as W1, b1, W2, b2:

* **Weights (W1, W2):** These are matrices of numbers that define the "connection strength" between neurons. These are the *most important* numbers. **"Learning" is the process of slowly adjusting these weights.**  
* **Biases (b1, b2):** These are tiny adjustment knobs that help "nudge" the network's calculations in the right direction.

### **The Learning Loop: How the Model "Learns"**

The grad\_descent function is the main training loop. It repeats three steps over and over:

1. **Forward Propagation (Make a Guess)**  
2. **Backpropagation (Find the "Blame" for the Error)**  
3. **Gradient Descent (Update the Parameters)**

#### **1\. Forward Propagation: Making a Guess**

This is the forward\_prop function. It flows *forward* from the input pixels to the final prediction.

* **Step 1:** The input pixels (X) are multiplied by the first set of weights (W1).  
* **Step 2 (ReLU):** The result goes through the ReLU (Rectified Linear Unit) activation function. This is a simple math rule: **if a number is negative, make it 0; otherwise, let it pass.** This non-linear "switch" is what allows the network to learn complex patterns instead of just simple, straight lines.  
* **Step 3:** The result from the hidden layer (A1) is multiplied by the second set of weights (W2).  
* **Step 4 (Softmax):** The 10 final scores go through softmax. This brilliant function converts the raw scores (e.g., \[-2.1, 5.5, 0.1, ...\]) into probabilities that all add up to 1 (e.g., \[0.01, 0.89, 0.02, ...\]). The network's guess is the digit with the highest probability.

#### **2\. Backpropagation: Learning from Mistakes**

This is the back\_prop function and the "magic" of deep learning. It's the process of figuring out *how wrong* the guess was and *how to fix it*.

* **Step 1 (Find the Error):** We compare the network's prediction (the softmax probabilities) to the *true* label (which we converted to a one\_hot vector, e.g., \[0,0,0,0,0,1,0,0,0,0\] for a '5'). The difference between the two is the "error."  
* **Step 2 (Find the "Blame"):** Using calculus (specifically, the chain rule), backpropagation calculates the "gradient." The gradient (e.g., dW2) is a matrix of numbers that represents **how much each single weight in W2 was "to blame" for the error.**  
* **Step 3 (Propagate the "Blame"):** The error is then passed *backward* to the hidden layer, and the same process calculates how much W1 was to blame (dW1).

At the end of back\_prop, we have four gradient matrices: dW1, db1, dW2, db2. These matrices are our "instructions" for how to update our weights to be better.

#### **3\. Gradient Descent: Updating the Weights**

This is the update\_params function. It's the simplest, but most important, step. It just follows the instructions from backpropagation.

The new weight is the old weight, minus a small adjustment:  
W1 \= W1 \- alpha \* dW1

* alpha is the **Learning Rate**. It's a tiny number (like 0.10) that controls *how big* of a step we take. It's like a "caution" setting. We make small, careful updates to our weights so we don't "overshoot" the correct answer.

### **Putting It All Together**

The grad\_descent function simply runs these three steps in a for loop (e.g., 700 times).

1. **Guess** (Forward Prop)  
2. **Find Blame** (Backprop)  
3. **Update Weights** (Gradient Descent)

With each loop, the weights (W1, W2) get slightly "smarter." The accuracy we print shows this live: the model goes from guessing randomly (around 10% accuracy) to correctly identifying digits over 85% of the time, all by repeating this simple mathematical loop.