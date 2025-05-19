# Neural Network implementation
The story behind this project, is that last year I got very annoyed that I was hearing the buzzwords 'Neural Networks' and 'AI' non-stop, while I knew nothing about them.
So one day I decided to research the maths, write it down, then code an implementation from scratch, no TensorFlow no PyTorch, just numpy and pandas.

You can find the train and test datasets in [kaggle](https://www.kaggle.com/code/ermalbejko/digit-classifier-1?select=test.csv)
You will need to rename them to `nn_train.csv` and `nn_test.csv` in order for the python notebook to work, or you can simply change the code.

# Train and test
The train file has 42000 records, we use 41000 for training and 1000 for holdout data to measure performance on unseen data.

The test file has 28000 records, in the end of the notebook I have used the function `test_prediction` in a way that it recreates the image, showcases the prediction and the probability density function of the 10 digit clasess (in lament terms I wanted to see how the model was 'thinking' i.e. how confident it is in the answer).


# Neural Network Mathematical Formulation

## 1. Data Preparation

- Each input image is a vector $X \in \mathbb{R}^{784}$.
- Each label $Y$ is an integer (0–9), one-hot encoded for training.

---

## 2. Parameter Initialization

$$
\begin{aligned}
W_1 &\in \mathbb{R}^{196 \times 784} \\
b_1 &\in \mathbb{R}^{196 \times 1} \\
W_2 &\in \mathbb{R}^{10 \times 196} \\
b_2 &\in \mathbb{R}^{10 \times 1}
\end{aligned}
$$

---

## 3. Forward Propagation

$$
\begin{aligned}
Z_1 &= W_1 X + b_1 \\
A_1 &= \text{ReLU}(Z_1) = \max(0, Z_1) \\
Z_2 &= W_2 A_1 + b_2 \\
A_2 &= \text{softmax}(Z_2) = \frac{e^{Z_2}}{\sum e^{Z_2}}
\end{aligned}
$$

---

## 4. One-Hot Encoding

$$
Y_{\text{one-hot}}[i, y_i] = 1, \quad \text{else } 0
$$

---

## 5. Loss Function (Cross-Entropy)

$$
L = -\frac{1}{m} \sum_{i=1}^m \sum_{j=1}^{10} Y_{j}^{(i)} \log(A_{2,j}^{(i)})
$$

---

## 6. Backward Propagation

$$
\begin{aligned}
dZ_2 &= A_2 - Y_{\text{one-hot}} \\
dW_2 &= \frac{1}{m} dZ_2 A_1^T \\
db_2 &= \frac{1}{m} \sum dZ_2 \\
dZ_1 &= W_2^T dZ_2 \cdot \text{ReLU}'(Z_1) \\
dW_1 &= \frac{1}{m} dZ_1 X^T \\
db_1 &= \frac{1}{m} \sum dZ_1
\end{aligned}
$$

---

## 7. Parameter Update (Gradient Descent)

$$
\theta := \theta - \alpha \cdot d\theta
$$

Where $\alpha$ is the learning rate.

---

## 8. Activation Functions

- **ReLU**:
  $\text{ReLU}(z) = \max(0, z)$

- **Sigmoid**:
  $\sigma(z) = \frac{1}{1 + e^{-z}}$

- **Softmax**:
  $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$

---

## 9. Prediction and Accuracy

- **Prediction**:
  $\hat{y} = \arg\max(A_2)$

- **Accuracy**:
  $\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of samples}}$
