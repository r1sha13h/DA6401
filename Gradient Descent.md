# Gradient Descent and Its Variants
## DA6401: Introduction to Deep Learning - Module 2B

**Instructor:** Ganapathy Krishnamurthi  
**Department:** Data Science and Artificial Intelligence, IIT Madras

---

## Slide 1: The Learning Algorithm

### What is a Learning Algorithm?
- A systematic procedure to adjust model parameters to minimize a loss function
- Also known as **optimizers** in deep learning

---

## Slide 2: The Learning Algorithm - Mathematical Objective

### Mathematical Objective:
$$\theta^* = \arg \min \mathcal{L}(\theta) \quad (1)$$

**where:**
- $\theta$ represents all learnable parameters (weights and biases)
- $\mathcal{L}(\theta)$ is the loss function measuring prediction error
- $\theta^*$ is the optimal parameter set

---

## Slide 3: The Learning Algorithm - Common Learning Algorithms

### Common Learning Algorithms:
- Gradient Descent (GD)
- Stochastic Gradient Descent (SGD)
- Adam, RMSprop, AdaGrad, etc.

---

## Slide 4: Parameter Update - Learnable Parameters

### Learnable Parameters:
- **Weights:** $W^{(1)}, W^{(2)}, \ldots, W^{(L)}$
- **Biases:** $b^{(1)}, b^{(2)}, \ldots, b^{(L)}$
- **Collectively denoted as:** $\theta = \{W^{(1)}, b^{(1)}, \ldots, W^{(L)}, b^{(L)}\}$

---

## Slide 5: Parameter Update - Parameter Update Rule

### Parameter Update:
$$\theta_{new} = \theta_{old} + \Delta\theta \quad (2)$$

---

## Slide 6: Parameter Update - The Problem

### Problem:
Without proper scaling, we have no control over:
- How much the parameters change
- How fast the network learns
- Stability of training

---

## Slide 7: Parameter Update - Introducing Learning Rate

### Solution: 
Introduce a **learning rate** $\eta$:
$$\theta_{new} = \theta_{old} + \eta \cdot \Delta\theta \quad (3)$$

**where:** $0 < \eta < 1$ (typically $\eta \approx 0.001$ to $0.1$)

This scales the update to a **controlled step size**.

---

## Slide 8: Parameter Update - Vector Perspective Setup

### Setup:
**Parameter vector:** 
$$\theta = [w, b]$$

---

## Slide 9: Parameter Update - Vector Perspective Update Vector

### Update vector:
$$\Delta\theta = [\Delta w, \Delta b]$$

---

## Slide 10: Parameter Update - Vector Perspective Problem (Part 1)

### Problem: 
Directly adding $\Delta\theta$ to $\theta$ may cause unstable learning!

*Visual illustration showing parameter space with update vector being too large*

---

## Slide 11: Parameter Update - Vector Perspective Problem (Part 2)

### Problem continued:
When $\Delta\theta$ is too large, the update overshoots and can destabilize learning.

*Visual showing the uncontrolled update step in parameter space*

---

## Slide 12: Parameter Update - Vector Perspective Problem (Part 3)

### Problem continued:
The magnitude of $\Delta\theta$ can lead to updates that are "Too large!"

*Visual showing the parameter trajectory becoming unstable*

---

## Slide 13: Parameter Update - Vector Perspective Solution Introduction

### Solution:
Introduce **learning rate** $\eta$:
$$\theta_{new} = \theta + \eta \cdot \Delta\theta$$

**where:** $0 < \eta < 1$ (typically $\eta \approx 0.001$ to $0.1$)

This scales the update to a **controlled step size**

---

## Slide 14: Parameter Update - Vector Perspective with Learning Rate

### Solution (continued):
The learning rate $\eta$ ensures that we move in a controlled direction with a controlled magnitude.

*Visual showing parameter space with a scaled update vector pointing from origin*

---

## Slide 15: Parameter Update - Vector Perspective Illustration 2

### Solution with Learning Rate:
Shows the trajectory of parameters as they are updated with controlled learning rate steps.

*Visual showing dashed line representing the full $\Delta\theta$ and the actual scaled update*

---

## Slide 16: Parameter Update - Vector Perspective Illustration 3

### Solution Convergence:
The scaled updates $\eta \cdot \Delta\theta$ lead toward convergence.

*Visual showing the parameter reaching $\theta_{new}$ with a controlled green step*

---

## Slide 17: Deriving the Incremental Update - Taylor Series Expansion

### Taylor Series Expansion:
For a function $f(x)$ around point $a$:
$$f(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2 + \frac{f'''(a)}{3!}(x - a)^3 + \cdots \quad (4)$$

---

## Slide 18: Deriving the Incremental Update - Applying to Loss Function

### Applying to Loss Function:
Consider the loss after a parameter update: $\mathcal{L}(\theta + \eta \cdot \Delta\theta)$

**Taylor expansion around $\theta$:**
$$\mathcal{L}(\theta + \eta \cdot \Delta\theta) = \mathcal{L}(\theta) + \nabla_\theta \mathcal{L}(\theta)^T \cdot (\eta \cdot \Delta\theta)$$
$$+ \frac{1}{2}(\eta \cdot \Delta\theta)^T H(\theta)(\eta \cdot \Delta\theta) + O(\eta^3) \quad (5)$$

**where** $H(\theta)$ is the Hessian matrix (second derivatives)

---

## Slide 19: Deriving the Incremental Update - First-Order Approximation

### First-Order Approximation:
Since $\eta$ is typically small (e.g., 0.001), terms with $\eta^2, \eta^3, \ldots$ are negligible:
$$\mathcal{L}(\theta + \eta \cdot \Delta\theta) \approx \mathcal{L}(\theta) + \eta \cdot \nabla_\theta \mathcal{L}(\theta)^T \cdot \Delta\theta \quad (6)$$

---

## Slide 20: Deriving the Update Rule - Minimizing Loss

### Deriving the Update Rule:
To **minimize** the loss, we want:
$$\mathcal{L}(\theta + \eta \cdot \Delta\theta) < \mathcal{L}(\theta) \quad (7)$$

**This requires:**
$$\nabla_\theta \mathcal{L}(\theta)^T \cdot \Delta\theta < 0 \quad (8)$$

**Optimal choice:** Set $\Delta\theta = -\nabla_\theta \mathcal{L}(\theta)$

This gives the **steepest descent direction**!

---

## Slide 21: The Gradient Descent Update Rule - Key Principle

### Recall the Key Principle:
- The gradient $\nabla_\theta \mathcal{L}(\theta)$ points in the direction of **steepest increase** of the loss
- To minimize loss, we move in the **opposite direction**

---

## Slide 22: The Gradient Descent Update Rule - Formal Update

### Formal Update Rule:
$$\boxed{\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)} \quad (9)$$

**where:**
- $\theta_t$ = parameters at iteration $t$
- $\eta$ = learning rate (step size)
- $\nabla_\theta \mathcal{L}(\theta_t)$ = gradient of loss w.r.t. parameters
- The minus sign ensures we move **downhill**

---

## Slide 23: The Gradient Descent Update Rule - Component-wise

### Component-wise Update:
$$W_{t+1}^{(l)} = W_t^{(l)} - \eta \cdot \frac{\partial \mathcal{L}}{\partial W^{(l)}} \quad (10)$$

$$b_{t+1}^{(l)} = b_t^{(l)} - \eta \cdot \frac{\partial \mathcal{L}}{\partial b^{(l)}} \quad (11)$$

---

## Slide 24: The Complete Learning Process - Step 1: Forward Pass

### The Complete Learning Process

**Input Layer** → **Hidden Layer** → **Output Layer**

Example network with 2 inputs, 2 hidden units, 1 output.

### Step 1: Forward Pass
$$h_1 = \sigma(v_{11}x_1 + v_{21}x_2 + b_1)$$
$$h_2 = \sigma(v_{12}x_1 + v_{22}x_2 + b_2)$$
$$\hat{y} = \sigma(w_1 h_1 + w_2 h_2 + b_3)$$

---

## Slide 25: The Complete Learning Process - Step 2: Compute Loss

### Step 2: Compute Loss
$$\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$$

For this example: $\mathcal{L} = \frac{1}{2}(\hat{y} - 1)^2$

---

## Slide 26: The Complete Learning Process - Step 3: Backward Pass

### Step 3: Backward Pass (Backpropagation)

Compute: $\frac{\partial \mathcal{L}}{\partial v_{11}}, \frac{\partial \mathcal{L}}{\partial v_{21}}, \frac{\partial \mathcal{L}}{\partial w_1}, \frac{\partial \mathcal{L}}{\partial w_2}, \ldots$

**Gradients flow backward through the network**

---

## Slide 27: The Complete Learning Process - Step 4: Update Parameters

### Step 4: Update Parameters
$$w_1 \leftarrow w_1 - \eta \frac{\partial \mathcal{L}}{\partial w_1}, \quad w_2 \leftarrow w_2 - \eta \frac{\partial \mathcal{L}}{\partial w_2}$$

$$v_{ij} \leftarrow v_{ij} - \eta \frac{\partial \mathcal{L}}{\partial v_{ij}} \quad \text{for all } i,j$$

---

## Slide 28: The Complete Learning Process - Step 5: Evaluate on Validation Set

### Step 5: Evaluate on Validation Set
- Run forward pass on validation data $(x_i, y_i)$
- Compute validation loss $\mathcal{L}_{val}$
- Check for convergence or overfitting

---

## Slide 29: The Complete Learning Process - Iterate Until Convergence

### Repeat: Iterate Until Convergence

**For each epoch:**
- Repeat Steps 1-4 for all training samples
- Evaluate on validation set (Step 5)

**Stop when:**
- Loss converges or max epochs reached

---

## Slide 30: The Complete Learning Process - Algorithm

### Training Algorithm

**Input:** $\mathcal{D}_{train} = \{(x_i, y_i)\}_{i=1}^N$, learning rate $\eta$, epochs $E$

**Initialize:** $\theta_0 = \{v_{ij}, w_k, b_l\}$ randomly

**For each $e = 1$ to $E$:**
- **For each data sample $(x_i, y_i)$ in $\mathcal{D}_{train}$:**
  1. **Forward Pass:** $\hat{y}_i = f(x_i; \theta)$
  2. **Compute Loss:** $\mathcal{L} = \frac{1}{|batch|} \sum \ell(\hat{y}, y)$
  3. **Backward Pass:** $\nabla_\theta \mathcal{L}$ via backpropagation
  4. **Update:** $\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}$
- **End For**
- 5. **Evaluate:** Compute $\mathcal{L}_{val}$ on validation set
- 6. **If** converged or performance degrades: **break**
- **End For**

**Return:** Optimized parameters $\theta^*$

---

## Slide 31: Stochastic and Mini-Batch Gradient Descent

### Section: Stochastic and Mini-Batch Gradient Descent

Variations of gradient descent that differ in how many samples are used to compute gradients.

---

## Slide 32: Vanilla Gradient Descent (Batch Gradient Descent)

### Definition: 
Compute the gradient using the **entire training dataset** in each update step.

---

## Slide 33: Vanilla Gradient Descent - Mathematical Formulation

### Mathematical Formulation:
Given training dataset $\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^N$ with $N$ samples:

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f(x^{(i)}, \theta), y^{(i)}) \quad (12)$$

---

## Slide 34: Vanilla Gradient Descent - Gradient Computation

### Gradient Computation:
$$\nabla_\theta \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \nabla_\theta \ell(f(x^{(i)}, \theta), y^{(i)}) \quad (13)$$

---

## Slide 35: Vanilla Gradient Descent - Parameter Update

### Parameter Update:
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{N} \sum_{i=1}^N \nabla_\theta \ell(f(x^{(i)}, \theta_t), y^{(i)}) \quad (14)$$

---

## Slide 36: Vanilla Gradient Descent - Properties (Advantages)

### Advantages:
- Guaranteed convergence to global minimum for convex functions
- Stable gradient estimates (low variance)
- Smooth convergence trajectory
- Deterministic updates

---

## Slide 37: Vanilla Gradient Descent - Properties (Disadvantages)

### Disadvantages:
- **Computationally expensive** for large datasets
- Requires entire dataset to fit in memory
- Slow updates (one update per full pass through data)
- Cannot handle online learning scenarios

### Computational Complexity:
- Time per update: $O(N \cdot d)$ where $d$ = number of parameters
- Memory: $O(N)$ to store all training samples

---

## Slide 38: Stochastic Gradient Descent (SGD)

### Definition: 
Compute the gradient using **one random sample** at a time.

---

## Slide 39: Stochastic Gradient Descent - Mathematical Formulation

### Mathematical Formulation:
At iteration $t$, randomly select one sample $(x^{(t)}, y^{(t)})$ from $\mathcal{D}$:

$$\ell^{(t)}(\theta) = \ell(f(x^{(t)}, \theta), y^{(t)}) \quad (15)$$

---

## Slide 40: Stochastic Gradient Descent - Gradient Computation

### Gradient Computation:
$$\nabla_\theta \ell^{(t)}(\theta) = \nabla_\theta \ell(f(x^{(t)}, \theta), y^{(t)}) \quad (16)$$

---

## Slide 41: Stochastic Gradient Descent - Parameter Update

### Parameter Update:
$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \ell(f(x^{(t)}, \theta_t), y^{(t)}) \quad (17)$$

**Key Property:** The gradient from a single sample is an **unbiased estimator** of the true gradient:
$$\mathbb{E}_{i \sim \text{Uniform}(1,N)} [\nabla_\theta \ell^{(t)}(\theta)] = \nabla_\theta \mathcal{L}(\theta) \quad (18)$$

---

## Slide 42: Stochastic Gradient Descent - Properties (Advantages)

### Advantages:
- **Fast updates** - one update per sample
- Low memory requirements
- Can escape shallow local minima due to noise
- Suitable for online learning
- Can handle large datasets that don't fit in memory

---

## Slide 43: Stochastic Gradient Descent - Properties (Disadvantages)

### Disadvantages:
- **High variance** in gradient estimates
- Noisy convergence trajectory
- May not converge exactly to minimum (oscillates around it)
- Requires careful learning rate tuning

---

## Slide 44: Stochastic Gradient Descent - Properties (Computational Complexity)

### Computational Complexity:
- Time per update: $O(d)$ where $d$ = number of parameters
- Memory: $O(1)$ only one sample at a time
- Total updates per epoch: $N$ (one per sample)

---

## Slide 45: Mini-batch Gradient Descent

### Definition: 
Compute the gradient using a **small batch of samples** (compromise between Batch GD and SGD).

---

## Slide 46: Mini-batch Gradient Descent - Mathematical Formulation

### Mathematical Formulation:
At iteration $t$, randomly select a mini-batch $B_t \subset \mathcal{D}$ with $|B_t| = b$ samples:

$$B_t = \{(x^{(i)}, y^{(i)})\}_{i \in \mathcal{I}_t}, \quad |B_t| = b \quad (19)$$

---

## Slide 47: Mini-batch Gradient Descent - Gradient Computation

### Gradient Computation:
$$\nabla_\theta \mathcal{L}_{B_t}(\theta) = \frac{1}{b} \sum_{i \in B_t} \nabla_\theta \ell(f(x^{(i)}, \theta), y^{(i)}) \quad (20)$$

---

## Slide 48: Mini-batch Gradient Descent - Parameter Update

### Parameter Update:
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{b} \sum_{i \in B_t} \nabla_\theta \ell(f(x^{(i)}, \theta_t), y^{(i)}) \quad (21)$$

**Common batch sizes:** $b \in \{16, 32, 64, 128, 256, 512\}$

---

## Slide 49: Mini-batch Gradient Descent - Properties (Advantages)

### Advantages:
- **Best of both worlds**: balance between stability and speed
- Reduced variance compared to SGD
- Efficient use of vectorized operations (GPU acceleration)
- More stable convergence than SGD
- Faster than Batch GD

---

## Slide 50: Mini-batch Gradient Descent - Properties (Disadvantages)

### Disadvantages:
- Requires tuning of batch size hyperparameter
- Still has some noise (though less than SGD)

---

## Slide 51: Mini-batch Gradient Descent - Properties (Computational Complexity)

### Computational Complexity:
- Time per update: $O(b \cdot d)$ where $b$ = batch size, $d$ = parameters
- Memory: $O(b)$ to store mini-batch
- Updates per epoch: $\lceil N/b \rceil$

---

## Slide 52: Comparison - Gradient Descent Variants

| Property | Batch GD | SGD | Mini-batch GD |
|----------|----------|-----|---------------|
| Samples per update | $N$ | $1$ | $b$ |
| Updates per epoch | $1$ | $N$ | $N/b$ |
| Gradient variance | Low | High | Medium |
| Convergence speed | Slow | Fast | Fast |
| Memory requirement | High | Low | Medium |
| Stability | High | Low | Medium |
| GPU efficiency | Low | Low | High |

### Gradient Estimate Variance:
$$\text{Var}[\nabla_\theta \mathcal{L}] = \begin{cases} 0 & \text{Batch GD} \\ \sigma^2 & \text{SGD} \\ \sigma^2/b & \text{Mini-batch GD} \end{cases} \quad (22)$$

where $\sigma^2$ is the variance of individual sample gradients.

---

## Slide 53: Key Terminologies - Iteration

### 1. Iteration:
- One forward pass + backward pass + parameter update
- One step of gradient descent

---

## Slide 54: Key Terminologies - Epoch

### 2. Epoch:
- One complete pass through the **entire training dataset**
- All $N$ training samples have been seen once by the network

---

## Slide 55: Key Terminologies - Batch Size

### 3. Batch Size ($b$):
- Number of samples processed before one parameter update
- Determines memory usage and gradient variance

---

## Slide 56: Key Terminologies - When Does the Network Learn?

### 4. When does the network learn?
- The network learns (parameters update) at **each iteration**
- Learning = parameter update via gradient descent

---

## Slide 57: Key Terminologies - Iterations per Epoch

### Number of iterations in one epoch:

**Batch Gradient Descent:**
$$\text{Iterations per epoch} = 1 \quad (23)$$
- Uses all $N$ samples in one iteration

**Stochastic Gradient Descent:**
$$\text{Iterations per epoch} = N \quad (24)$$
- Uses 1 sample per iteration, needs $N$ iterations to see all data

**Mini-batch Gradient Descent:**
$$\text{Iterations per epoch} = \lceil N/b \rceil \quad (25)$$
- Uses $b$ samples per iteration, needs $\lceil N/b \rceil$ iterations

---

## Slide 58: Key Terminologies - Concrete Example

### Example: $N = 1000$ training samples

| Algorithm | Batch Size | Iterations/Epoch | Updates/Epoch |
|-----------|-----------|------------------|---------------|
| Batch GD | $b = 1000$ | 1 | 1 |
| Mini-batch GD | $b = 100$ | 10 | 10 |
| Mini-batch GD | $b = 50$ | 20 | 20 |
| Mini-batch GD | $b = 32$ | 32 | 32 |
| SGD | $b = 1$ | 1000 | 1000 |

**Key Observation:**
- More iterations per epoch → more frequent updates
- SGD: 1000 updates per epoch vs Batch GD: 1 update per epoch
- Mini-batch strikes a balance

---

## Slide 59: Refined Learning Algorithm - Batch Gradient Descent

### Batch Gradient Descent Algorithm

**Input:** $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, learning rate $\eta$, epochs $E$

**Initialize:** $\theta_0$ randomly

**For epoch $e = 1$ to $E$:**
- // One iteration per epoch
- 1. **Forward Pass:** Compute $\hat{y}^{(i)} = f(x^{(i)}; \theta)$ for all $i = 1, \ldots, N$
- 2. **Compute Loss:** $\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(\hat{y}^{(i)}, y^{(i)})$
- 3. **Backward Pass:** $\nabla_\theta \mathcal{L} = \frac{1}{N} \sum_{i=1}^N \nabla_\theta \ell(\hat{y}^{(i)}, y^{(i)})$
- 4. **Update:** $\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}$
- 5. **Evaluate:** Compute $\mathcal{L}_{val}$ on validation set
- 6. **If** converged or performance degrades: **break**
- **End For**

**Return:** Optimized parameters $\theta^*$

---

## Slide 60: Refined Learning Algorithm - Stochastic Gradient Descent

### Stochastic Gradient Descent Algorithm

**Input:** $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, learning rate $\eta$, epochs $E$

**Initialize:** $\theta_0$ randomly

**For epoch $e = 1$ to $E$:**
- Randomly shuffle $\mathcal{D}$
- **For** $i = 1$ to $N$: // $N$ iterations per epoch
  - 1. **Forward Pass:** $\hat{y}^{(i)} = f(x^{(i)}; \theta)$
  - 2. **Compute Loss:** $\ell^{(i)} = \ell(\hat{y}^{(i)}, y^{(i)})$
  - 3. **Backward Pass:** $\nabla_\theta \ell^{(i)} = \nabla_\theta \ell(\hat{y}^{(i)}, y^{(i)})$
  - 4. **Update:** $\theta \leftarrow \theta - \eta \cdot \nabla_\theta \ell^{(i)}$
- **End For**
- 5. **Evaluate:** Compute $\mathcal{L}_{val}$ on validation set
- **End For**

**Return:** Optimized parameters $\theta^*$

---

## Slide 61: Refined Learning Algorithm - Mini-batch Gradient Descent

### Mini-batch Gradient Descent Algorithm

**Input:** $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, learning rate $\eta$, batch size $b$, epochs $E$

**Initialize:** $\theta_0$ randomly

**For epoch $e = 1$ to $E$:**
- Randomly shuffle $\mathcal{D}$
- **For** $t = 1$ to $\lceil N/b \rceil$: // $\lceil N/b \rceil$ iterations per epoch
  - $B_t = \{(x^{(i)}, y^{(i)})\}_{i=(t-1)b+1}^{\min(tb, N)}$ // Create mini-batch
  - 1. **Forward Pass:** $\hat{y}^{(i)} = f(x^{(i)}; \theta)$ for all $(x^{(i)}, y^{(i)}) \in B_t$
  - 2. **Compute Loss:** $\mathcal{L}_{B_t} = \frac{1}{|B_t|} \sum_{(x,y) \in B_t} \ell(\hat{y}, y)$
  - 3. **Backward Pass:** $\nabla_\theta \mathcal{L}_{B_t} = \frac{1}{|B_t|} \sum_{(x,y) \in B_t} \nabla_\theta \ell(\hat{y}, y)$
  - 4. **Update:** $\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}_{B_t}$
- **End For**
- 5. **Evaluate:** Compute $\mathcal{L}_{val}$ on validation set
- **End For**

**Return:** Optimized parameters $\theta^*$

---

## Summary

This module covers the complete fundamentals of **Gradient Descent** and its variants:

- **Learning Algorithms** and loss minimization objectives
- **Parameter Updates** with controlled learning rates
- **Mathematical derivation** using Taylor series expansion
- **The Gradient Descent Update Rule** in scalar and component-wise forms
- **Complete Learning Process** with forward/backward passes and validation
- **Three main variants:**
  - **Batch GD:** Stable but slow on large datasets
  - **Stochastic GD:** Fast updates but high variance
  - **Mini-batch GD:** Best balance for practical deep learning
- **Key terminologies:** iterations, epochs, batch size, and their relationships
- **Refined algorithms** with detailed pseudocode for each variant

The gradient descent algorithm and its variants form the foundation for all modern deep learning optimization techniques.
