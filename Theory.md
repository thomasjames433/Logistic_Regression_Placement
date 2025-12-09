# Logistic Regression Notes

## Part 1: Theory and Cost Function Derivation

**Topic:** Logistic Regression / Classification (0/1)

**1. Sigmoid Function:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
*(Includes a sketch of the S-curve/Sigmoid function where $z=0 \rightarrow 0.5$)*

**2. Hypothesis & Likelihood:**
* **Parameters:** $\theta = [\theta_0, \theta_1, \dots]$
* **Hypothesis:** $h_\theta(x) = \sigma(\theta^T x)$
* **Likelihood Function:**
$$L(\theta) = \prod_{i=1}^{m} [h_\theta(x^{(i)})]^{y^{(i)}} [1 - h_\theta(x^{(i)})]^{1-y^{(i)}}$$
    * Where $y^{(i)}$ is the class label (0 or 1).
    * If $y=1$, we want the first term to be large. If $y=0$, we want the second term to be large.

**3. Log-Likelihood & Cost Function:**
* We need a negative function (to minimize).
* Likelihood product is numerically unstable (very small numbers), so we take the **log**.
* To make it scalable, we divide by $m$.

**The Cross-Entropy Loss Function (Cost Function):**
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \ln(h_\theta(x^{(i)})) + (1-y^{(i)}) \ln(1-h_\theta(x^{(i)})) \right]$$
* Goal: Minimize $J(\theta)$

---

## Part 2: Gradient Descent Derivation

**Chain Rule Application:**
To find the gradient $\frac{\partial J}{\partial \theta_j}$, we use the chain rule:
$$\frac{\partial J}{\partial \theta_j} = \frac{\partial J}{\partial L_{(i)}} \cdot \frac{\partial L_{(i)}}{\partial h_\theta} \cdot \frac{\partial h_\theta}{\partial z} \cdot \frac{\partial z}{\partial \theta_j}$$

**Derivatives of components:**
1.  **Loss Derivative:** $\frac{y^{(i)}}{h_\theta(x^{(i)})} - \frac{1-y^{(i)}}{1-h_\theta(x^{(i)})}$
2.  **Sigmoid Derivative:** $\sigma(z)(1-\sigma(z)) = h(1-h)$
3.  **Linear Derivative:** $x_j$

**Simplification:**
When you multiply these terms, the denominators cancel out, leaving:
$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

**Vectorized Gradient:**
$$\nabla J(\theta) = \frac{1}{m} X^T (h - y)$$

**Update Rule:**
$$\theta := \theta - \alpha \nabla J(\theta)$$

**Example Scenario Defined:**
* **3 Features:**
    1.  Hours of study
    2.  Hours of sleep
    3.  Previous exam score
* **Prediction:** Pass or Fail

---

## Part 3: Numerical Example

**Problem Setup:**
* **$m = 3$** (3 students to train)
* **Model Parameters ($\theta$):**
$$
\theta = \begin{bmatrix} 
-4 \\\\ 
0.04 \\\\ 
0.3 \\\\ 
0.02 
\end{bmatrix} 
\begin{aligned} 
& \leftarrow \text{Intercept} \\\\ 
& \leftarrow \text{Hours of Study} \\\\ 
& \leftarrow \text{Hours of Sleep} \\\\ 
& \leftarrow \text{Previous Test Score} 
\end{aligned}
$$

**Input Matrix ($X$):**
Columns: $[1, \text{Study}, \text{Sleep}, \text{Score}]$
$$
X = \begin{bmatrix} 
1 & 5 & 7 & 60 \\\\ 
1 & 2 & 6 & 55 \\\\ 
1 & 8 & 5 & 80 
\end{bmatrix}
$$

**Target Labels ($y$):**
$$
y = \begin{bmatrix} 
1 \\\\ 
0 \\\\ 
1 
\end{bmatrix}
$$

**Step 1: Compute Linear Model ($z = X\theta$)**
$$z^{(1)} = 1(-4) + 5(0.04) + 7(0.3) + 60(0.02) = -0.5$$
$$z^{(2)} = -1.02$$
$$z^{(3)} = -0.58$$

**Step 2: Compute Sigmoid ($h = \frac{1}{1+e^{-z}}$)**
$$h^{(1)} = 0.3775$$
$$h^{(2)} = 0.264$$
$$h^{(3)} = 0.359$$

**Step 3: Calculate Error Vector ($h - y$)**
$$
h - y = \begin{bmatrix} 
0.3775 - 1 \\\\ 
0.264 - 0 \\\\ 
0.359 - 1 
\end{bmatrix} 
= \begin{bmatrix} 
-0.6225 \\\\ 
0.264 \\\\ 
-0.641 
\end{bmatrix}
$$

**Step 4: Compute Gradient ($\nabla J(\theta)$)**
Formula: $\frac{1}{m} X^T (h-y)$

$$\nabla J(\theta) = \frac{1}{3} \begin{bmatrix} -0.9995 \\ -7.7125 \\ -5.9788 \\ -74.011 \end{bmatrix}$$