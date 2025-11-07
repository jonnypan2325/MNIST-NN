---
layout: default
title: "Week 1: Linear Classifiers"
subtitle: "Understanding how weights and bias create decision boundaries."
---
# Week 1: Linear Classifiers


## Accompanying notebook:
<a href="https://colab.research.google.com/drive/199srarZ--a0UNndEN86Kv-Q321S5Z8UZ#scrollTo=20b1d82c" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open the Complementary Notebook In Colab">
</a>

Click File -> Save a copy in Drive

## Summary:
In this module, you will:
- Learn the difference between classification and regression.
- Understand the geometric and mathematical intuition behind a linear classifier.
- Discover how weights and bias define a decision boundary.
- See why some problems cannot be solved with a simple linear model.


## Classification vs Regression
You've probably heard the terms 'Artificial Intelligence' and 'Machine Learning' everywhere. While they sound complex, they are built on fundamental concepts. Let's start with the basics.

In machine learning, we often solve two main types of problems: **Classification** and **Regression**.

Think of **Classification** as sorting things into labeled bins. For example, an email filter classifies incoming messages as either "Spam" or "Not Spam." The output is a distinct category.

**Regression**, on the other hand, is about predicting a continuous number. A weather app predicting that tomorrow's temperature will be 86°F is a regression task.

In this MNIST tutorial, we will focus entirely on classification. Before we move onto the actual Neural Networks, we want to understand what classification is using the simplest model that performs it: a **linear classifier**.

In essence, a linear classifier is a model that makes decisions by drawing a straight line (or a flat plane in higher dimensions) to separate different categories.

Imagine you have a scatter plot with a red dot and a blue dot. A linear classifier finds the best straight line to place between them, creating a "decision boundary." Everything on one side of the line is classified as blue, and everything on the other is classified as red.


## Geometric Intuition

The power of a linear classifier comes from its simplicity. The separating line it creates is called a **decision boundary**. By changing the line's slope and intercept, we can move it around to best fit the data.

A dataset is called **linearly separable** if you can draw a single straight line that perfectly separates all the data points into their correct categories. In our case, we have two distinct points that don't overlap, so our dataset is linearly seperable. 

Later on, we'll see that this assumption doesn't hold for larger or more complex datasets.



## Mathematical Perspective

How does a computer understand a line? Through an equation. The decision boundary of a linear classifier is simply the equation of a line.

You might remember from algebra that a line can be written in point-slope form as $y = mx + c$. While this is useful, it's not the most convenient for classification. Let's rearrange it:

$$
y - mx - c = 0
$$

By moving all terms to one side, we've established a new rule: for any point $(x, y)$ that lies exactly on the line, the expression $y - mx - c$ will be equal to zero.

In machine learning, we use a slightly different notation to make this more general. We represent our input coordinates as $(x_1, x_2)$ instead of $(x, y)$. Let's rewrite the equation again:

$$
(-m)x_1 + (1)x_2 - c = 0
$$

This is the same equation, just with different variable names. Now, let's map this to machine learning terms:
- We'll call `-m` our first **weight**, $w_1$.
- We'll call `1` our second **weight**, $w_2$.
- We'll call `-c` our **bias**, $b$.

Substituting these gives us the standard machine learning form for a line:

$$
w_1 x_1 + w_2 x_2 + b = 0
$$

But why do we now have two weights ($w_1$, $w_2$) when the original equation only had one slope ($m$)?

The two weights $w_1$ and $w_2$ work together to define the slope of the line. Specifically, the slope is $-w_1 / w_2$. This more general form is powerful because it can represent *any* line, including vertical lines (which $y = mx + c$ cannot, as the slope would be infinite). It also scales to higher dimensions for more complex problems, which is essential for machine learning.

This equation defines the decision boundary. The expression $w_1 x_1 + w_2 x_2 + b$ does something very useful:
-   For any point on one side of the line, its value is **positive**.
-   For any point on the other side, its value is **negative**.
-   For any point exactly *on* the line, its value is **zero**.

For instance, let's take a simple line that goes through the origin (0,0) and (1,1). In slope-intercept form, this is $y = x$. To convert this to our general form, we move all terms to one side: $y - x = 0$.

Now, let's match this to our notation. We replace $x$ with $x_1$ and $y$ with $x_2$:

$$x_2 - x_1 = 0$$

Or, to match the $w_1x_1 + w_2x_2 + b = 0$ structure, we can write it as:

$$(-1)x_1 + (1)x_2 + 0 = 0$$

This gives us the parameters for the line $y = x$:
-   $w_1 = -1$
-   $w_2 = 1$
-   $b = 0$

Now, let's use these parameters to classify some new points. Are the following points ***above***, ***below***, or ***on*** the line?
-   (3, 1)
-   (1, 3)
-   (3, 3)

Try plugging them into the expression $(-1) \cdot x_1 + (1) \cdot x_2 + 0 = ?$ and check the sign of the result.

<details markdown="1">
<summary style="font-size: 1.2em;"> ==> <strong>Click for the solution</strong></summary>

Let's test each point:

-   **Point (3, 1):**
    -   $x_1 = 3$, $x_2 = 1$
    -   Expression: $(-1 * 3) + (1 * 1) + 0 = -3 + 1 = -2$
    -   The result is **negative**, so this point is *below* the line $y = x$.

-   **Point (1, 3):**
    -   $x_1 = 1$, $x_2 = 3$
    -   Expression: $(-1 * 1) + (1 * 3) + 0 = -1 + 3 = 2$
    -   The result is **positive**, so this point is *above* the line $y = x$.

-   **Point (3, 3):**
    -   $x_1 = 3$, $x_2 = 3$
    -   Expression: $(-1 * 3) + (1 * 3) + 0 = -3 + 3 = 0$
    -   The result is **zero**, so this point is exactly *on* the line $y = x$.

</details>
Checking the sign of the expression is central to our classification rule, which can also be written as:

$$
y_{predicted} = \text{sign}(w \cdot x + b)
$$

Let's break down what each symbol means:
-   $y_{predicted}$: The predicted class of the point (in our code, negative maps to 0, positive maps to 1).
-   $w$: The **weights** ($w_1$, $w_2$, etc.), which define the orientation (slope) of the decision boundary.
-   $x$: The **features** of the input data point (e.g., its coordinates $(x_1, x_2)$).
-   $b$: The **bias**, which shifts the decision boundary, acting like an intercept.
-   $sign()$: A function that returns `+1` if the result is positive and `-1` if it's negative, effectively sorting the point into one of two classes.

## The "Learning" in Machine Learning

The weights and bias are the heart of a linear classifier. They are the **learnable parameters** of the model. When we say a machine "learns," we mean it is systematically adjusting its weights and bias to find the optimal decision boundary that correctly classifies the training data.

Initially, the weights and bias might be set to random values, resulting in a line that poorly separates the data. The goal of a training algorithm (like the Perceptron, which we'll see next) is to iteratively tweak these parameters until the line correctly divides the categories. This process of adjustment is the essence of learning in this context.


## Limitations of Linear Models

As you'll discover in the notebook, linear classifiers fail when faced with nonlinear patterns. Datasets like the `XOR` pattern or concentric circles (half-moons) cannot be separated by a single straight line. This limitation is what motivates the need for more powerful models.

## What's Next

In Week 2, we’ll see how the perceptron learns to adjust its weights automatically, forming the foundation of neural networks. This will allow us to solve the nonlinear problems that linear classifiers can't handle.