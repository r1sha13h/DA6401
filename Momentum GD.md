### Limitations of Vanilla Gradient Descent

**Ravines and Slow Convergence**
- A ravine is a region where the loss surface curves much more steeply in one dimension than another
- Vanilla GD oscillates across the narrow dimension while crawling slowly along the bottom
- Results in inefficient **zigzag patterns** instead of a direct path to the minimum

![Vanilla GD Convergence](Vanilla_GD_Convergence.png)

*Visualizing the Problem: Vanilla GD in Ravines - Zigzag path, slow progress, many oscillations. Vanilla GD makes slow progress toward optimum, wasting steps on oscillations perpendicular to the optimal direction.*

**Oscillations Around Minima**
- When gradient directions change rapidly, GD bounces back and forth near the minimum
- This wastes computation and prevents smooth, stable convergence

**No Acceleration Mechanism**
- Each update is **independent** — there is no memory of past gradient directions
- Update rule: $\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$
- For Vanilla GD The update magnitude depends *only* on the current gradient, so GD:
  - Cannot speed up along directions where the gradient consistently points the same way
  - Cannot dampen movement in directions where it keeps oscillating

**Core Insight**
All three problems stem from the same root cause — GD has no way to **accumulate velocity** in favorable directions or **suppress oscillations** in unfavorable ones. This is the key motivation for momentum-based optimizers like SGD with Momentum and Adam.

---

## Momentum-based Gradient Descent

### Update Rule

**Momentum Update Rule:**

**Momentum Gradient Descent**
Initialize velocity: $v_0 = 0$
At each iteration t:

$$v_{t+1} = \gamma v_t + \eta \nabla_\theta \mathcal{L}(\theta_t)$$

$$\theta_{t+1} = \theta_t - v_{t+1}$$

**Parameters:**
- $\gamma \in [0, 1)$: Momentum coefficient (typically 0.9 or 0.99)
- $\eta > 0$: Learning rate
- $v_t$: Velocity (accumulated gradient)

**Special Cases:**
- $\gamma = 0$: Reduces to vanilla gradient descent
- $\gamma \to 1$: Maximum momentum, very smooth trajectory

### Momentum: Dampening Oscillations

Consider oscillating gradients:
Suppose gradients alternate: $g_1 = +5, g_2 = -4, g_3 = +5, g_4 = -4, ...$

**Vanilla GD:** (with $\eta = 0.1$)
$\Delta \theta_1 = -0.5,$
$\Delta \theta_2 = +0.4,$
$\Delta \theta_3 = -0.5,$
$\Delta \theta_4 = +0.4$

Large oscillations persist!

**Momentum GD:** (with $\eta = 0.1, \gamma = 0.9$)
$v_1 = 0.9(0) + 0.1(5) = 0.5$

$v_2 = 0.9(0.5) + 0.1(-4) = 0.45 - 0.4 = 0.05$

$v_3 = 0.9(0.05) + 0.1(5) = 0.045 + 0.5 = 0.545$

$v_4 = 0.9(0.545) + 0.1(-4) = 0.49 - 0.4 = 0.09$

Oscillations dampened! Velocity stabilizes around 0.

### Understanding Momentum: Exponential Moving Average

**Claim:** Velocity $v_t$ is an exponential moving average (EMA) of all past gradients.

**Proof:** Expand the recursion $v_{t+1} = \gamma v_t + \eta \nabla_\theta \mathcal{L}(\theta_t)$

Starting from $v_0 = 0$:
$v_1 = \gamma v_0 + \eta g_0 = \eta g_0$

$v_2 = \gamma v_1 + \eta g_1 = \gamma \eta g_0 + \eta g_1$

$v_t = \eta \sum_{i=0}^{t-1} \gamma^{t-1-i} g_i$

where $g_i = \nabla_\theta \mathcal{L}(\theta_i)$

**General Form:**

$$v_t = \eta \sum_{i=0}^{t-1} \gamma^{t-1-i} \nabla_\theta \mathcal{L}(\theta_i)$$

### Understanding Momentum: Weighted Sum of Gradients

$$v_t = \eta (g_{t-1} + \gamma g_{t-2} + \gamma^2 g_{t-3} + \gamma^3 g_{t-4} + \cdots)$$

Recent gradients have weight $\gamma^0 = 1$ (most influence)
Gradients decay exponentially with age: $\gamma^1, \gamma^2, \gamma^3, ...$

**With $\gamma = 0.9$:**
- $g_{t-1}$: weight = 1.0
- $g_{t-2}$: weight = 0.9
- $g_{t-3}$: weight = 0.81
- $g_{t-10}$: weight = 0.35

**Interpretation:** Momentum smooths out gradient noise by averaging over history!

### Momentum vs Vanilla GD: Visual Comparison

![Momentum vs Vanilla GD](Momentum_vs_VanillaGD.png)

**Vanilla GD:** Many oscillations, Zigzag pattern, slow convergence

**Momentum:** Smooth, direct path, faster convergence

### Momentum-based Gradient Descent: Complete Algorithm

**Momentum Gradient Descent**
**Input:** $D = \{(x^{(i)}, y^{(i)})\}_{i=1}^N$, learning rate $\eta$, momentum $\gamma$, epochs $E$, batch size $b$

**Initialize:** $\theta_0$ randomly, $v_0 = 0$

**For epoch $e = 1$ to $E$:**
1. Shuffle $D$
2. **For each mini-batch $B_t$:**
   1. Forward: $\hat{y} = f(x; \theta)$ for $(x, y) \in B_t$
   2. Loss: $L_{B_t} = \frac{1}{|B_t|} \sum_{(x,y) \in B_t} \ell(\hat{y}, y)$
   3. Backward: $g_t = \nabla_\theta L_{B_t}$
   4. Update velocity: $v_{t+1} = \gamma v_t + \eta g_t$
   5. Update parameters: $\theta \leftarrow \theta - v_{t+1}$
3. Evaluate on validation set

**Return:** $\theta^*$

### Practical Considerations

**Choosing Momentum Coefficient $\gamma$:**
- $\gamma = 0.9$: Standard choice, good for most problems
- $\gamma = 0.95$ or $0.99$: More aggressive, useful for very noisy gradients
- $\gamma < 0.9$: Less common, reduces momentum effect

**Learning Rate with Momentum:**
- **Adjustment needed**: Learning rate typically needs to be reduced compared to vanilla GD
- **Effective step size**: Momentum amplifies the actual step taken during optimization
- **Practical guideline**: If using learning rate $\eta$ with vanilla GD, try $\eta/2$ or $\eta/3$ with momentum

**When to Use Momentum:**
Always recommended for deep neural networks
Particularly effective for:
- High-dimensional optimization
- Noisy gradients (mini-batch SGD)
- Ill-conditioned loss surfaces (ravines, plateaus)

### Summary: Momentum-based Gradient Descent

**Key Ideas**
1. Accumulates velocity: $v_{t+1} = \gamma v_t + \eta g_t$
2. Exponential moving average of gradients
3. Dampens oscillations, accelerates in consistent directions
4. Converges faster than vanilla GD
5. Note: Both Stochastic and Mini-batch variants of momentum GD exist

---

## Numerical Comparison: Momentum vs Vanilla GD

![Momentum vs Vanilla GD Numerical](Momentum_vs_VanillaGD_Numerical.png)

### Setup

$$L(\theta_1, \theta_2) = 5\theta_1^2 + 0.1\theta_2^2, \quad \nabla L = [10\theta_1,\ 0.2\theta_2]$$
$$\eta = 0.22,\quad \gamma = 0.90,\quad \text{Start: } \theta_1 = 1.5,\ \theta_2 = 5.0$$

### Vanilla GD — Step by Step

**Update rule:** $\theta_{t+1} = \theta_t - \eta \cdot \nabla L(\theta_t)$

| Step | θ₁ | θ₂ | g₁ = 10θ₁ | g₂ = 0.2θ₂ | Δθ₁ = −η·g₁ | Δθ₂ = −η·g₂ | Loss |
|------|------|------|-----------|-----------|------------|------------|------|
| 0 | 1.5000 | 5.0000 | 15.000 | 1.000 | −3.300 | −0.220 | 13.75 |
| 1 | −1.8000 | 4.7800 | −18.000 | 0.956 | +3.960 | −0.210 | 18.48 |
| 2 | +2.1600 | 4.5697 | +21.600 | 0.914 | −4.752 | −0.201 | 25.42 |
| 3 | −2.5920 | 4.3686 | −25.920 | 0.874 | +5.702 | −0.192 | 35.50 |
| 5 | −3.7325 | 3.9926 | −37.325 | 0.799 | +8.212 | −0.176 | 71.25 |
| 8 | +6.4497 | 3.4885 | +64.497 | 0.698 | −14.189 | −0.154 | **209.21** 💥 |
| 11 | −11.145 | 3.0479 | −111.45 | 0.610 | +24.519 | −0.134 | **622.0** 💥 |

**Key observation:** θ₁ multiplies by $1 - \eta \cdot 10 = -1.2$ every step → flips sign AND grows by ×1.2. Loss explodes. θ₂ barely moves.

### Momentum GD — Step by Step

**Update rule:** $v_{t+1} = \gamma \cdot v_t + \eta \cdot \nabla L(\theta_t)$, then $\theta_{t+1} = \theta_t - v_{t+1}$

| Step | θ₁ | θ₂ | g₁ | g₂ | v₁ (prev) | v₂ (prev) | v₁ (new) | v₂ (new) | Loss |
|------|------|------|------|------|-----------|-----------|---------|---------|------|
| 0 | 1.5000 | 5.0000 | 15.000 | 1.000 | 0.000 | 0.000 | **3.300** | **0.220** | 13.75 |
| 1 | −1.8000 | 4.7800 | −18.000 | 0.956 | +3.300 | 0.220 | **−0.990** | **0.408** | 18.48 |
| 2 | −0.8100 | 4.3717 | −8.100 | 0.874 | −0.990 | 0.408 | **−2.673** | **0.560** | 5.19 |
| 3 | +1.8630 | 3.8118 | +18.630 | 0.762 | −2.673 | 0.560 | **+1.693** | **0.672** | 18.81 |
| 4 | +0.1701 | 3.1403 | +1.701 | 0.628 | +1.693 | 0.672 | **+1.898** | **0.743** | 1.13 |
| 5 | −1.7277 | 2.3977 | −17.277 | 0.480 | +1.898 | 0.743 | **−2.093** | **0.774** ← peak | 15.50 |
| 8 | −0.7623 | 0.1272 | −7.623 | 0.025 | +2.208 | 0.729 | **+0.310** | **0.662** | 2.91 |
| 11 | +0.6626 | −1.572 | +6.626 | −0.314 | +0.345 | 0.466 | **+1.768** | **0.350** | **2.44** ✅ |

### The Dampening Mechanism — Step 1 in Detail

At step 1, θ₁ = −1.8, so the gradient **flips sign**: g₁ = −18 (negative).
But the velocity carried from step 0 is **v₁ = +3.3** (positive):

$$v_1^{\text{new}} = \underbrace{0.9 \times (+3.3)}_{\text{memory }= +2.97} + \underbrace{0.22 \times (-18)}_{\text{gradient }= -3.96} = -0.99$$

The opposing memory term **cancels most of the gradient** → actual step in θ₁ is only **−(−0.99) = +0.99**, vs Vanilla GD's full step of **+3.96**. That is a **4× smaller update** — this is the dampening.

### The Acceleration Mechanism — θ₂ Velocity Buildup

Since g₂ is consistently positive (θ₂ > 0 for the first several steps), each velocity update **adds onto the previous**:

$$v_2: 0 \to 0.220 \to 0.408 \to 0.560 \to 0.672 \to 0.743 \to \mathbf{0.774} \text{ (peak at step 5)}$$

Compare VGD which takes a fixed step of $\eta \cdot g_2 \approx 0.22$ — Momentum's effective step grows to **3.5× larger** at peak. That is the acceleration.

---

## Nesterov Accelerated Gradient (NAG)

### Limitations of Momentum-based GD

**The Problem**: Momentum builds up velocity but cannot anticipate turns, leading to:
- Excessive overshooting past the minimum
- Unnecessary oscillations (U-turns)
- Slower convergence near the minimum

**Mathematical Analysis**:
At iteration t, momentum GD computes:
$$v_{t+1} = \gamma v_t + \eta \nabla_\theta \mathcal{L}(\theta_t)$$

**The Issue**: Gradient $\nabla_\theta \mathcal{L}(\theta_t)$ is evaluated at current position $\theta_t$, but the update moves in direction $v_{t+1}$ which includes momentum from $v_t$. No "look-ahead" to see where we're actually going.

**Consequence**: When approaching minimum from one direction with high velocity, the algorithm overshoots and must correct course in the next iteration.

### Intuition behind NAG

**Key Idea**: "Look ahead" before computing the gradient

**NAG Philosophy**:
1. First, take a step in the direction of accumulated momentum: $\theta_t - \gamma v_t$
2. Then, compute gradient at this "look-ahead" position
3. Make a correction based on this anticipated gradient

### Momentum GD vs NAG

**Momentum GD**: Gradient at $\theta_t$ (blind to future)

**NAG**: Gradient at look-ahead (anticipates future)

**Result**: NAG corrects the trajectory before overshooting, leading to:
- Fewer oscillations
- Smoother convergence
- Better performance near the minimum

### Update Rule for NAG

**Nesterov Accelerated Gradient (Original Formulation)**:
$$v_{t+1} = \gamma v_t + \eta \nabla_\theta \mathcal{L}(\theta_t - \gamma v_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

**where**:
- $\theta_t - \gamma v_t$ is the "look-ahead" position
- $\gamma \in [0, 1)$ is the momentum coefficient (typically 0.9)
- $\eta > 0$ is the learning rate

**Key Difference from Momentum GD**:
- **Momentum**: $\nabla_\theta \mathcal{L}(\theta_t)$ (gradient at current position)
- **NAG**: $\nabla_\theta \mathcal{L}(\theta_t - \gamma v_t)$ (gradient at look-ahead)

### Mathematical Comparison

![Momentum vs NAG](Momentum_vs_NAG.png)

| Method | Update Rule |
|---|---|
| **Vanilla GD** | $\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$ |
| **Momentum GD** | $v_{t+1} = \gamma v_t + \eta \nabla_\theta \mathcal{L}(\theta_t)$<br>$\theta_{t+1} = \theta_t - v_{t+1}$ |
| **NAG** | $v_{t+1} = \gamma v_t + \eta \nabla_\theta \mathcal{L}(\theta_t - \gamma v_t)$<br>$\theta_{t+1} = \theta_t - v_{t+1}$ |

**Computational Cost**: Same as momentum GD: one gradient evaluation per iteration. Slight overhead: computing $\theta_t - \gamma v_t$ before gradient. Negligible compared to gradient computation cost.

### Advantages of NAG

**Compared to Vanilla GD**:
- Faster convergence: $O(1/t^2)$ vs $O(1/t)$
- Better handling of ill-conditioned problems: Accumulates velocity in consistent directions
- Reduced sensitivity to learning rate: Momentum provides stability

**Compared to Momentum GD**:
- Reduced oscillations: Look-ahead mechanism anticipates turns
- Better convergence near minimum: Less overshooting
- Theoretically optimal: Proven optimal convergence rate for first-order methods
- More responsive: Adapts to changing gradient directions faster

**Practical Benefits**:
- Same computational cost as momentum GD
- Simple to implement (one-line change from momentum)
- Works well in practice for deep learning
- Particularly effective for problems with elongated valleys and high condition number optimization landscapes

**Empirical Observation**: In machine learning applications, NAG often converges in 30-50% fewer iterations than standard momentum GD while using the same hyperparameters.

### Algorithm: Nesterov Accelerated Gradient (NAG)

**Nesterov Accelerated Gradient**
**Input**: $D = \{(x^{(i)}, y^{(i)})\}_{i=1}^N$, learning rate $\eta$, momentum $\gamma$, epochs $E$, batch size $b$

**Initialize**: $\theta_0$ randomly, $v_0 = 0$

**For epoch $e = 1$ to $E$:**
1. Shuffle $D$
2. **For each mini-batch $B_t$:**
   1. Look-ahead: $\tilde{\theta} = \theta - \gamma v_t$
   2. Forward: $\hat{y} = f(x; \tilde{\theta})$ for $(x, y) \in B_t$
   3. Loss: $L_{B_t} = \frac{1}{|B_t|} \sum_{(x,y) \in B_t} \ell(\hat{y}, y)$
   4. Backward: $g_t = \nabla_{\tilde{\theta}} L_{B_t}$
   5. Update velocity: $v_{t+1} = \gamma v_t + \eta g_t$
   6. Update parameters: $\theta \leftarrow \theta - v_{t+1}$
3. Evaluate on validation set

**Return**: $\theta^*$

### Practical Considerations

**When to use NAG**:
- Optimization problems with ravines or valleys
- When momentum GD shows oscillatory behavior
- When faster convergence is desired with minimal tuning

**Hyperparameter tuning**:
- Start with $\gamma = 0.9, \eta = 0.01$
- Increase $\gamma$ (up to 0.99) for smoother optimization surfaces
- Decrease $\gamma$ if oscillations persist
- Use learning rate schedules for best performance

### NAG: Limitation on Sparse & Dense Features

**Root cause:** NAG still uses a single global $\eta$ — it only improves *direction* (look-ahead), never *per-weight step size*.

---

**Sparse Features:**
- Gradient arrives rarely $\to$ velocity decays to $\approx 0$ between appearances
- When gradient finally arrives, $\gamma v_t \approx 0$, so look-ahead $\tilde{\theta} \approx \theta_t$
- NAG reduces to **Vanilla GD** for sparse weights — no benefit

**Dense Features:**
- Velocity accumulates to $\frac{\eta g}{1-\gamma}$ — up to **10× the base step** at steady state
- Cannot slow down near convergence — overshoots for well-trained dense weights

**The Dilemma:**

$$\text{Small } \eta \Rightarrow \text{sparse features starve} \quad | \quad \text{Large } \eta \Rightarrow \text{dense features explode}$$

No single $\eta$ satisfies both simultaneously.

---

**What's Missing:**
NAG needs a **per-parameter denominator** that automatically scales each weight's step by its gradient history — large steps for sparse weights, small steps for dense ones. This is the motivation for **AdaGrad** $\to$ **RMSProp** $\to$ **Adam**.

### Summary: Nesterov Accelerated Gradient

**Key Takeaways**
1. NAG improves momentum GD by computing gradients at "look-ahead" positions
2. Achieves optimal $O(1/t^2)$ convergence for smooth convex functions
3. Reduces oscillations and overshooting near minima
4. Same computational cost as momentum GD
5. Both Stochastic and Mini-batch variants exist

**The NAG Principle**: "Anticipate where you're going, then correct course accordingly"

---

## The Learning Rate Dilemma

### Standard Update Rule (SGD)

**Standard Update Rule (SGD)**:
$$w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial L}{\partial w}$$

### The Core Problem: Single Learning Rate for All Parameters

When you do any gradient update — Vanilla GD, Momentum, NAG — the rule always looks like:

$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

That **same scalar $\eta$** multiplies every gradient, for every weight, at every step. This assumes all weights deserve equally-sized updates — which is almost never true.

### Sparse vs Dense Features

**Dense feature** — appears in almost every training example.
- Example: a "length of sentence" feature in NLP, or a common word like *"the"*
- Its gradient is updated **every single batch** → accumulates large, reliable signal
- Needs a **small η** — already getting plenty of updates, don't overshoot

**Sparse feature** — appears rarely across training examples.
- Example: a rare word like *"photosynthesis"* in a vocabulary embedding
- Its gradient is **non-zero only occasionally** → gets very few updates
- Needs a **large η** — must make the most of each rare update it gets

### Concrete Example: Text Classification Network

Imagine a text classification network with a vocabulary of 50,000 words:

| Weight | How often updated | What happens with fixed η |
|---|---|---|
| Embedding for *"the"* | Every batch (dense) | Overfit / overshoots — too many large steps |
| Embedding for *"photosynthesis"* | Once in 1000 batches (sparse) | Barely learns — each rare update is too small |
| Bias term | Every batch | Fine with medium η |

With a single $\eta = 0.01$, the word *"the"* might converge just fine, but *"photosynthesis"* almost **never gets a meaningful update**. If you raise $\eta$ to help sparse features, you blow up the dense ones.

### Why Momentum & NAG Don't Solve This

Momentum and NAG **accumulate velocity** — but they still scale every weight by the same $\eta$. They help with the *direction* problem (ravines, oscillations) but do nothing about the *magnitude* problem (sparse vs dense). A sparse feature's velocity never builds up meaningfully because it's zero most of the time, so momentum gives it no advantage.

### Additional Challenges: Loss Landscape

The loss surface is not a perfect bowl. It has cliffs, valleys, and plateaus. A rate that works at the start (steep area) might be terrible at the end (flat area).

**High η**: The model overshoots the minimum and may diverge (explode)

**Low η**: The model learns painfully slowly and gets stuck in local minima

### The Solution: Per-Parameter Learning Rates

The solution is to give **each weight its own effective learning rate**, automatically scaled based on its update history:

$$\theta_{t+1}^{(i)} = \theta_t^{(i)} - \frac{\eta}{\text{(something based on past gradients of weight } i\text{)}} \cdot g_t^{(i)}$$

- Weights with **large, frequent gradients** → denominator grows large → effective η shrinks
- Weights with **small, rare gradients** → denominator stays small → effective η stays large

This is exactly what **AdaGrad** does (accumulates squared gradients), and what **Adam** refines further. The single scalar $\eta$ becomes a *base* learning rate, and the actual per-weight step size is adapted automatically.

### Next Steps: Adaptive Algorithms

We need algorithms that adapt the learning rate automatically for each parameter.

**RMSProp**: Adjusts rate based on recent gradient magnitude.
**Adam**: Combines both Momentum and RMSProp.

---

## Testing Code for GD, Momentum GD, and NAG

```python
import numpy as np

delLoss = lambda x, y: np.array([10*x, 0.2*y])
neta = 0.01
gamma = 0.9
theta = [np.array([1.5, 5.0])]
v = np.zeros(2)  # velocity initialized to zero

for i in range(300):
    theta_arg = theta[-1] - gamma*v # NAG
    #theta_arg = theta[-1]  # Momentum
    grad = delLoss(*theta_arg)
    v = gamma * v + neta * grad # Momentum
    #v = neta * grad    # VGD
    newTheta = theta[-1] - v
    theta.append(newTheta)

for t in theta:
    print(t)
```

**Usage Instructions:**
- **Vanilla GD**: Uncomment `v = neta * grad` and comment out `v = gamma * v + neta * grad`
- **Momentum GD**: Use `theta_arg = theta[-1]` and `v = gamma * v + neta * grad`
- **NAG**: Use `theta_arg = theta[-1] - gamma*v` and `v = gamma * v + neta * grad`