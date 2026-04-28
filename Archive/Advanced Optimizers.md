# Advanced Optimizers

## Fixed Learning Rate Optimizers
- **Same learning rate for all parameters**: This is suboptimal because neural network parameters have varying gradient scales. Some features update frequently with large gradients (dense features), while others have infrequent small updates (sparse features). A uniform fixed rate either overshoots active parameters (causing instability) or underupdates sparse ones (slowing progress).
- **Difficult to choose learning rate**: 
  - Too large: Leads to divergence (unstable training).
  - Too small: Results in slow convergence.
- **Manual learning rate schedules**: Require expert knowledge and problem-specific tuning, making it less efficient and more error-prone.

## Adaptive Learning Rates
Adapt the learning rate for each parameter individually based on:
- History of gradients
- Frequency of updates
- Magnitude of updates 

### Methods Covered:
- AdaGrad
- RMSProp
- AdaDelta
- Adam

## AdaGrad Optimizer
- **Algorithm**: Adaptive Gradient Algorithm (Duchi et al., 2011)
- **Key Innovation**:
  - Accumulates squared gradients
  - **Main Principle**: Parameters with large gradients should have their learning rate reduced, while parameters with small gradients should have their learning rate increased.
- **Ideal For**:
  - Sparse data (e.g., NLP Tasks)
  - Features with different frequencies
- **Main Principle**: Parameters with large gradients should have their learning rate reduced, while parameters with small gradients should have their learning rate increased.

**Example (Word Embedding Model)**:
  - Word "the" appears frequently → Large gradients accumulated → Reduce learning rate (already learned well).
  - Word "aardvark" appears rarely → Small gradients accumulated → Increase learning rate (needs more learning).

**General Principle**:
  - Frequently updated parameters → Smaller effective learning rate.
  - Infrequently updated parameters → Larger effective learning rate.
  - Automatically balances the learning process.

### AdaGrad Algorithm:
- **Initialization**:
  - $\theta_0$ (parameters), learning rate $\eta$, small constant $\epsilon$ (e.g., $10^{-8}$)
  - Gradient accumulator $G_0 = 0$
- **Algorithm Steps**:
  1. Compute gradient: $g_t = \nabla_{\theta} J(\theta_t)$
  2. Accumulate squared gradient: $G_t = G_{t-1} + g_t \odot g_t$ (element-wise)
  3. Update parameters: $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$
- **Key Points**:
  - $G_t$: Accumulates sum of squared gradients (element-wise).
  - $\odot$: Element-wise multiplication.
  - $\epsilon$: Prevents division by zero.
  - Effective learning rate: $\frac{\eta}{\sqrt{G_t + \epsilon}}$ (decreases over time).

### Analyzing AdaGrad:
- **Advantages**:
  - **Automatic Learning Rate Adaptation**: No manual tuning required; each parameter gets its own learning rate based on gradient history
  - **Sparse Feature Optimization**: Excellent for sparse data where parameters need larger effective learning rates
  - **Dimension-wise Scaling**: Automatically scales learning rate for each dimension, reducing oscillations in steep directions while maintaining progress in flat directions
  - **Theoretical Guarantees**: Proven convergence properties and automatic per-parameter adaptation
- **Observations & Limitations**:
![LR decay illustration](./LR%20Decay.png)
  - **Critical Problem: Aggressive Learning Rate Decay**
    - $G_t = \sum_{\tau=1}^{t} g_{\tau}^{2}$ grows monotonically without bound
    - Effective learning rate $\frac{\eta}{\sqrt{G_t + \epsilon}} \to 0$ as $t \to \infty$
    - Learning rate becomes infinitesimally small, training stops
  - **Consequences**:
    - Early gradients have lasting impact (all past gradients have equal weight)
    - Cannot escape local minima late in training
    - Learning rate monotonically decreases, never recovers

## RMSProp Optimizer
- **Origin**: Proposed by Geoffrey Hinton (Coursera, 2012).
- **Key Idea**: Maintain an exponential moving average (EMA) of squared gradients to address AdaGrad's limitations. This approach:
  - Gives recent gradients more influence than old ones
  - Prevents aggressive learning rate decay
  - Keeps learning rates non-zero for continued learning
  - Adapts to recent gradient behavior
  - Better suited for non-convex and online learning settings

### Intuition
- Early training often has large, noisy gradients; later training has smaller, stable gradients.
- AdaGrad permanently reduces learning rates based on early gradients; RMSProp gradually “forgets” old gradients via EMA, focusing on recent behavior.
- **EMA of squared gradients**:
  - $E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) g_t^2$
  - Typical $\beta \approx 0.9$. Recent gradients get weight $(1-\beta)$, older gradients decay by $\beta$ each step. Roughly the past $\frac{1}{1-\beta}$ steps remain influential.

### Algorithm
1. **Initialize**: parameters $\theta_0$, base learning rate $\eta$, decay $\beta$ (≈0.9), moving average $E[g^2]_0 = 0$, small constant $\epsilon = 10^{-8}$.
2. **Iterate for each step $t$**:
   1. Compute gradient $g_t = \nabla_{\theta} J(\theta_t)$.
   2. Update EMA: $E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta)(g_t \odot g_t)$.
   3. Parameter update: $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \odot g_t$.
- **Key difference from AdaGrad**: AdaGrad sums all squared gradients ($G_t = G_{t-1} + g_t^2$), causing monotonic decay; RMSProp bounds the moving average through $\beta$, so learning rates can stabilize instead of vanishing.

### Comparison: AdaGrad vs RMSProp
- **Learning rate behavior**: AdaGrad only decreases; RMSProp can increase or decrease depending on recent gradients.
- **Long training**: AdaGrad often stops learning; RMSProp continues learning by maintaining adaptive rates.
- **Gradient memory**: AdaGrad treats all history equally; RMSProp emphasizes recent gradients via EMA.
- **Hyperparameters**: AdaGrad mainly needs $\eta$; RMSProp introduces $\beta$ alongside $\eta$.

### Limitations
- **Learning rate sensitivity**: RMSProp still depends heavily on the global learning rate $\eta$—too large causes instability/divergence, too small slows convergence, so manual tuning is required.
- **Unit inconsistency**: Update step $\Delta \theta = \frac{\eta}{\sqrt{E[g^2]_t}} g_t$ can have units that do not match the parameter scale, which may complicate optimization.
- **Extra hyperparameters**: Both $\eta$ and $\beta$ must be tuned, and their optimal values vary across problems.

### Decay Factor $\beta$ (a.k.a. $\gamma$)
- Controls the trade-off between remembering past squared gradients versus reacting to the latest gradient.
- Cache update: $E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) g_t^2$.
- $\beta E[g^2]_{t-1}$ is the memory term; $(1-\beta) g_t^2$ is the immediate sensitivity term.
- Higher $\beta$ (close to 1) emphasizes history and smooths updates; lower $\beta$ reacts sharply to current gradients.

  | $\beta$ (decay factor) | Combined Behavior & Characteristics | Problem |
  | --- | --- | --- |
  | $0$ | Only current gradient → jumps erratically, unstable with no memory | Too reactive |
  | Low (e.g., $0.5$) | Fast, volatile cache → highly reactive, noisy updates with frequent overshoot | Unstable |
  | $0.9$ (default) | ~10-step window → stable, adapts well (recommended baseline) | ✅ Sweet spot |
  | $\approx 1.0$ (e.g., $0.99$) | Very long memory → shrinks slowly, approaches AdaGrad behaviour (over-smoothed) | Over-smoothed |
  | $1$ | Frozen at $E[g^2]_0 = 0$ → $\frac{\eta}{\epsilon}$ (huge), explodes immediately | **Catastrophic** |

> **Critical Warning**: $\beta = 1$ is the worst possible setting — the denominator never adapts, and division by $\epsilon$ turns every gradient step into a catastrophically large update. In practice, $\beta$ is always kept strictly less than 1.

### Why $\beta$ fixes AdaGrad
- AdaGrad sums all squared gradients, so $G_t$ grows indefinitely and learning rates shrink toward zero.
- RMSProp exponentially decays old gradients: a gradient from $k$ steps ago is weighted by $\beta^k (1-\beta)$; as $k$ grows, its influence becomes negligible.
- This keeps $E[g^2]_t$ bounded and allows the effective learning rate $\frac{\eta}{\sqrt{E[g^2]_t + \epsilon}}$ to recover in flat regions.

- **Steep regions**: gradients are large, $E[g^2]_t$ swells, effective LR shrinks → prevents overshoot.
- **Flat regions**: gradients are small, historical large gradients decay via $\beta^k$, effective LR increases → maintains progress.
- Net effect: LR shrinks in steep directions and self-corrects upward in flat zones, unlike AdaGrad.

### Unrolling the recursion & gradient fade-out
- Recursively expanding the cache: $E[g^2]_t = (1-\beta) g_t^2 + \beta(1-\beta) g_{t-1}^2 + \beta^2(1-\beta) g_{t-2}^2 + \dots$

- Gradient from $k$ steps ago has weight $\beta^k (1-\beta)$ (geometric decay).

  **Weight table for $\beta = 0.9$**

  | Steps ago $k$ | Weight $0.9^k \times 0.1$ | % contribution |
  | --- | --- | --- |
  | 0 | 0.1000 | 10.0% |
  | 1 | 0.0900 | 9.0% |
  | 5 | 0.0590 | 5.9% |
  | 10 | 0.0349 | 3.5% |
  | 20 | 0.0122 | 1.2% |
  | 50 | 0.0005 | ~0.05% |

  Historical gradients vanish quickly; by 50 steps their weight is essentially zero.

### Effective memory window vs $\beta$

Approximate window $\approx \frac{1}{1-\beta}$ steps.

  | $\beta$ | Eff. window | Weight @ $k=5$ | Weight @ $k=20$ | Behavior |
  | --- | --- | --- | --- | --- |
  | 0.5 | ~2 steps | $0.5^5 \times 0.5 \approx 0.016$ | ≈0 | Forgets extremely fast; highly reactive |
  | 0.9 | ~10 steps | 0.059 | 0.012 | Balanced default |
  | 0.99 | ~100 steps | 0.048 | 0.016 | Long memory; smooth but slow |

### Non-monotonic effective learning rate
- AdaGrad’s $G_t$ only increases, so $\eta_{\text{eff}} = \frac{\eta}{\sqrt{G_t + \epsilon}}$ only decreases.
- RMSProp’s $E[g^2]_t$ can rise or fall because of decay, so $\eta_{\text{eff}}$ is non-monotonic: it shrinks in steep bursts and recovers in flatter stretches.
- Numeric illustration ($\beta=0.9, \eta=0.1$):

  | Step | $g_t$ | $E[g^2]_t$ | $\eta_{\text{eff}} = \frac{0.1}{\sqrt{E[g^2]_t}}$ | Trend |
  | --- | --- | --- | --- | --- |
  | 1 | 5.0 | 2.50 | 0.063 | ↓ |
  | 2 | 5.0 | 4.75 | 0.046 | ↓ |
  | 3 | 0.5 | 4.30 | 0.048 | ↑ |
  | 4 | 0.5 | 3.895 | 0.051 | ↑ |
  | 5 | 0.5 | 3.53 | 0.053 | ↑ |

- This non-monotonicity helps navigate non-convex landscapes: shrink steps across cliffs, grow them in plateaus/saddles.

### Convergence despite LR recovery
- Actual parameter step is $\Delta \theta_t = \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$.
- Even if $\eta_{\text{eff}}$ rises near minima, $g_t$ simultaneously shrinks, so $\Delta \theta_t \to 0$.
- Example ($\eta=0.1, \beta=0.9$):

    | Step | $g_t$ | $E[g^2]_t$ | $\eta_{\text{eff}}$ | Actual step $= \eta_{\text{eff}} \cdot g_t$ |
    | --- | --- | --- | --- | --- |
    | 1 | 1.000 | 0.100 | 0.316 | 0.316 |
    | 2 | 0.500 | 0.115 | 0.295 | 0.147 |
    | 3 | 0.100 | 0.1045 | 0.309 | 0.031 |
    | 4 | 0.010 | 0.0941 | 0.326 | 0.003 |
    | 5 | 0.001 | 0.0847 | 0.344 | 0.0003 |

- Eventually $g_t^2 \ll \epsilon$, the denominator is dominated by $\epsilon$, and steps become directly proportional to $g_t$, ensuring convergence.

#### Why the Actual Step Still Vanishes
Near the true minimum, gradients become consistently tiny across many steps. The cache $E[g^2]_t$ follows with a lag but also shrinks. Eventually, when $g_t^2 \ll \epsilon$, the $\epsilon$ term dominates the denominator, and:
$$\Delta \theta_t \approx \frac{\eta g_t}{\sqrt{\epsilon}} \propto g_t \to 0$$
So $\epsilon$ is the actual convergence anchor — once gradients are tiny enough, the denominator stops shrinking further and the step size becomes purely proportional to the gradient. This is why $\epsilon$ is not just a numerical-stability trick; it plays a structural role in ensuring convergence.

## AdaDelta Optimizer
### Origin
- Matthew D. Zeiler, 2012: "ADADELTA: An Adaptive Learning Rate Method"

### Key Innovation
- No learning rate parameter needed - uses only exponential decay rates
- Corrects unit mismatch in RMSProp by accumulating both gradients and updates ($\Delta \theta$)
- Uses RMS of updates to scale learning for automatic unit correction
- More robust to hyperparameters

### Unit Inconsistency Problem and AdaDelta Solution
**Why RMSProp is "Inconsistent"**
RMSProp scales the gradient by the square root of the moving average of squared gradients. The update rule is:
$$\Delta \theta_t = -\frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

**The Units of RMSProp**


- The gradient $g_t$ represents the change in loss over the change in parameters, so its units are $[\theta]/[L]$.

- The denominator $\sqrt{E[g^2]_t}$ is the root mean square of the gradient, so it also carries units of $[\theta]/[L]$.

- Dividing them cancels the units entirely:
  $\frac{[\theta]/[L]}{[\theta]/[L]} = 1$ (unitless)

Therefore, the units of the update $\Delta \theta_t$ are simply the units of the learning rate $\eta$. The update has no relationship to the units of the weight parameters. If your parameters $\theta$ are in "kilograms" and your learning rate is just a scalar, you are effectively trying to update kilograms with a unitless number.

**Scale Cancellation Problem:** RMSProp's normalization creates scale-blindness. Imagine your gradient $g$ is 1. If you scale your weights by 10, your gradient $g$ also becomes 10. The RMSProp update yields:
- Original: $\Delta \theta = \frac{\eta}{\sqrt{1^2}} \cdot 1 = \eta$
- Scaled by 10: $\Delta \theta = \frac{\eta}{\sqrt{(10g)^2}} \cdot (10g) = \eta$

The result is the same! Even though your weights are now 10× larger, RMSProp still takes the exact same "step size" ($\eta$).

**[!IMPORTANT]**
This is a problem because if your parameters are huge, a tiny step of size $\eta$ will take forever to reach the minimum. You would have to manually go back and increase your learning rate ($\eta$) to 10× the original value to compensate.

**How AdaDelta Solves Both Problems**
AdaDelta replaces the fixed $\eta$ with a moving average of past updates (RMS[$\Delta \theta$]). The update rule is:
$$\Delta \theta_t = -\frac{\text{RMS}[\Delta \theta]_{t-1}}{\text{RMS}[g]_t} g_t$$

**Unit Consistency:** $\text{Units} = \dfrac{[\theta]}{[L]/[\theta]} \cdot \dfrac{[L]}{[\theta]} = [\theta]$. By including the RMS of previous updates $\text{RMS}[\Delta\theta]$ (units: $[\theta]$) in the numerator to cancel the $[L]/[\theta]$ denominator from $\text{RMS}[g_t]$, AdaDelta ensures the final update has the correct dimensions $[\theta]$, achieving unit consistency — entirely without needing a manually set $\eta$.

**Scale Adaptation:** The numerator acts as a "memory" of the scale of the weights. In the same 10× scaling scenario:
- If your weights are 10× larger, your gradients $g$ are 10× larger
- The denominator (RMS[$g$]) scales by 10
- Crucially, because your weights are larger, your previous updates ($\Delta \theta$) have also become 10× larger over time. So the numerator (RMS[$\Delta \theta$]) also scales by 10

$$\Delta \theta_{\text{new}} = \frac{10 \cdot \text{RMS}[\Delta \theta]}{10 \cdot \text{RMS}[g]} \cdot (10g) = 10 \cdot \Delta \theta_{\text{old}}$$

AdaDelta "sees" that the parameters are on a larger scale and automatically increases the step size to match. You don't have to touch a single hyperparameter.

### Update Rule
**Algorithm:**
1. Initialize: $\theta_0$, decay rate $\rho$ (typically 0.9 or 0.95)
2. Initialize: $E[g^2]_0 = 0$, $E[\Delta \theta^2]_0 = 0$, $\epsilon = 10^{-6}$
3. For $t = 1$ to $T$:
   - Compute gradient: $g_t = \nabla_\theta J(\theta_t)$
   - Accumulate gradient: $E[g^2]_t = \rho E[g^2]_{t-1} + (1-\rho) g_t \odot g_t$
   - Compute update: $\Delta \theta_t = -\frac{\sqrt{E[\Delta \theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} \odot g_t$
   - Accumulate updates: $E[\Delta \theta^2]_t = \rho E[\Delta \theta^2]_{t-1} + (1-\rho) \Delta \theta_t \odot \Delta \theta_t$
   - Update parameters: $\theta_{t+1} = \theta_t + \Delta \theta_t$

**Key Points:**
- No learning rate $\eta$! Only decay rate $\rho$
- Maintains two exponential moving averages: gradients and updates
- RMS = Root Mean Square: $\text{RMS}[x]_t = \sqrt{E[x^2]_t + \epsilon}$

### Advantages
- No manual learning rate tuning required
- Robust to choice of $\rho$ (typically 0.9-0.95 works well)
- Continues learning even after many iterations
- Unit-consistent updates

### Summary Comparison

<img src="Compare%20Convergence.png" alt="Compare Convergence" width="600"/>

| Method | Key Idea | Advantages | Limitations |
| --- | --- | --- | --- |
| AdaGrad | Accumulate all squared gradients | Per-parameter LR, Great for sparse data | LR → 0 and Stops learning |
| RMSProp | Exponential moving average of squared gradients | Continues learning, Non-convex friendly | Needs LR tuning and Unit mismatch |
| AdaDelta | Uses RMS of updates instead of LR | No LR needed, Unit consistent | Sometimes slower, One more hyperparameter |

## Adam Optimizer
### Origin
- **Adam** (Adaptive Moment Estimation): Kingma & Ba, 2015: "Adam: A Method for Stochastic Optimization"

### Key Innovation & Why Adam?
- Combines Momentum (first moment) + RMSProp (second moment) with bias correction for moments
- Most popular optimizer in deep learning - works well with minimal tuning
- Efficient computation and good default choice
- Bias correction especially important early in training
- Combines momentum direction with adaptive scaling

### Components
**Two Key Components:**

1. **First Moment (Mean) - Momentum:**
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
   - Exponential moving average of gradients
   - Provides direction and velocity
   - Typical value: $\beta_1 = 0.9$

2. **Second Moment (Variance) - Adaptive LR:**
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
   - Exponential moving average of squared gradients
   - Controls step size per parameter
   - Typical value: $\beta_2 = 0.999$

**Intuition:**
- $m_t$: "Where should we go?" (direction with momentum)
- $v_t$: "How big should our steps be?" (adaptive scaling)

### Bias Correction: Why It's Critical
<img src="Moment%20Estimation.png" alt="Moment Estimation" width="600"/>

**The Root Cause: Zero Initialization**
Adam initializes both moving averages at zero — $m_0 = 0$ and $v_0 = 0$. This is the **"original sin"** that creates bias. The moving average is a weighted sum of all past gradients, but at the start, one of those "past values" is the hard-coded zero, which artificially pulls estimates downward.

**Mathematical Derivation:**
Unrolling the first moment recursion $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$:
$$m_1 = (1-\beta_1)g_1$$
$$m_2 = \beta_1(1-\beta_1)g_1 + (1-\beta_1)g_2$$
$$m_3 = \beta_1^2(1-\beta_1)g_1 + \beta_1(1-\beta_1)g_2 + (1-\beta_1)g_3$$

General closed form: $m_t = (1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i$

General closed form: $v_t = (1-\beta_2)\sum_{i=1}^{t}\beta_2^{t-i}g_i^2$

Taking expected value (assuming i.i.d. gradients):
$$E[m_t] = E[g_t] \cdot (1 - \beta_1^t)$$
$$E[v_t] = E[g_t^2] \cdot (1 - \beta_2^t)$$

Since $(1-\beta_1^t) < 1$ always, both are **biased estimators** pulled toward zero. The same holds for $v_t$ with $\beta_2$.

**Why Biases Don't Cancel**

An intuitive (but incorrect) assumption is that since both $m_t$ and $v_t$ are underestimated, they might cancel in the update ratio. They do not, for two reasons:

*Reason 1: The Square Root Shield*
At $t=1$, the uncorrected update:
$$\text{Step} = \frac{(1-\beta_1)g}{\sqrt{(1-\beta_2)g^2}} = \frac{0.1g}{\sqrt{0.001}|g|} \approx 3.16$$
- Numerator $m_1$ is $\frac{1}{10}$ of true value
- Denominator $\sqrt{v_1}$ is $\frac{1}{31.6}$ of true value — the square root **softens** the underestimation of $v_t$, but not enough to match the numerator's underestimation
- Result: $\frac{1/10}{1/31.6} = 3.16$ — the step is still **3× larger than intended**

*Reason 2: Different Decay Rates ($\beta_1$ vs $\beta_2$)*
$\beta_1 = 0.9$ is deliberately chosen smaller so momentum adapts quickly to gradient direction. $\beta_2 = 0.999$ is deliberately large for a stable, long-term scale estimate. This asymmetry means the biases vanish at very different speeds:

| Iteration | $\beta_1^t = 0.9^t$ (m bias) | $\beta_2^t = 0.999^t$ (v bias) | Effect |
| --- | --- | --- | --- |
| 1 | 0.9 | 0.999 | Both heavily biased |
| 10 | 0.349 | 0.990 | $m_t$ recovering, $v_t$ still tiny |
| 50 | 0.005 | 0.951 | $m_t$ ~warmed up, $v_t$ still heavily biased |
| 100 | ≈0 | 0.905 | $m_t$ fully accurate, $v_t$ still lagging |
| 1000 | ≈0 | ≈0 | Both accurate |

Between iterations 10–100, the numerator recovers its true scale much faster than the denominator, causing a **mid-training spike** in effective learning rate — arguably more dangerous than the initial instability.

**The Correction Mechanism**
The bias-corrected estimates divide each moment by its remaining bias factor:
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

*At $t=1$:* Both corrections fully restore true values:
$$\hat{m}_1 = \frac{(1-\beta_1)g_1}{1-\beta_1} = g_1, \qquad \hat{v}_1 = \frac{(1-\beta_2)g_1^2}{1-\beta_2} = g_1^2$$

*At $t=100$:* Since $0.9^{100} \approx 0$ and $0.999^{100} \approx 0.905$, the corrections become:
$$\hat{m}_{100} \approx m_{100}, \qquad \hat{v}_{100} = \frac{v_{100}}{0.095}$$

Note that even at $t=100$, $\hat{v}_t$ still requires a non-trivial correction (dividing by 0.095 ≈ a **10.5× boost**), while $\hat{m}_t$ has already converged to its true value. This confirms that $v_t$'s bias lingers far longer and is the dominant source of instability without correction.

Once both $\beta_1^t$ and $\beta_2^t$ vanish (around $t \sim 1000$), both corrections become no-ops — they automatically phase themselves out.

**Impact on Stability**
Without bias correction, the effective step at $t=1$ with learning rate $\eta$ is:
$$\Delta\theta \approx \frac{\eta \cdot m_1}{\sqrt{v_1} + \epsilon} = \frac{\eta \cdot 0.1\,g}{\sqrt{0.001\,g^2}} \approx 3.16\,\eta \times \text{(gradient direction)}$$

The denominator $\sqrt{v_1}$ is **31.6× smaller** than its true value, while the numerator $m_1$ is only **10× smaller** — the denominator's collapse dominates, amplifying the step to 3× the intended learning rate during the most unstable initial phase.

With both corrections applied, the corrected update:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t$$

restores the ratio $\frac{\hat{m}_t}{\sqrt{\hat{v}_t}} \approx 1$ in terms of scale from the very first update, ensuring the **effective step size is approximately $\eta$ throughout training**. Without correcting $\hat{v}_t$ specifically, even a correctly scaled $\hat{m}_t$ would still produce an exploding denominator — the two corrections are inseparable for predictable training.

### Update Rule
**Algorithm:**
1. Initialize: $\theta_0$, stepsize $\alpha$ (default: 0.001)
2. Initialize: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$
3. Initialize: $m_0 = 0$, $v_0 = 0$
4. For $t = 1$ to $T$:
   - Compute gradient: $g_t = \nabla_\theta J(\theta_t)$
   - Update biased first moment: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
   - Update biased second moment: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
   - Compute bias-corrected first moment: $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$
   - Compute bias-corrected second moment: $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
   - Update parameters: $\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

**Default hyperparameters:** Learning rate $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

## NAdam Optimizer
### Origin
- Dozat, 2016: "Incorporating Nesterov Momentum into Adam"
- Full Name: Nesterov-accelerated Adam

### Key Innovation & Advantages
NAdam combines Adam's adaptive learning rates with Nesterov momentum to create a superior optimizer that leverages the best of both worlds. Adam provides per-parameter adaptive scaling while Nesterov momentum introduces look-ahead capability that computes gradients at anticipated positions rather than current positions.

This combination yields faster convergence than Adam, especially in computer vision tasks, while maintaining Adam's robustness and ease of use through superior gradient evaluation and momentum correction.

### Understanding Nesterov Momentum

**Key Insight:** Nesterov evaluates gradient at the anticipated position, allowing for better correction when overshooting.

### Update Rule
**Algorithm:**
1. Initialize: $\theta_0$, stepsize $\alpha$ (default: 0.002)
2. Initialize: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$
3. Initialize: $m_0 = 0$, $v_0 = 0$
4. For $t = 1$ to $T$:
   - Compute gradient: $g_t = \nabla_\theta J(\theta_t)$
   - Update first moment: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
   - Update second moment: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
   - Bias-corrected moments: $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$, $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$
   - Nesterov modification: $\bar{m}_t = \beta_1 \hat{m}_t + \frac{(1-\beta_1) g_t}{1-\beta_1^t}$
   - Update parameters: $\theta_{t+1} = \theta_t - \alpha \frac{\bar{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

### Convergence Adaptability
**The Core Problem with Adam's Momentum**
Adam uses classical momentum — it evaluates the gradient at the current position $\theta_t$, computes a moving average $m_t$ of past gradients, and steps in that accumulated direction. While effective, this creates an **overshoot problem**: when the optimizer is moving fast down a steep region and the minimum appears suddenly, Adam's momentum carries it past the minimum before it can react. It is reactive, not anticipatory — it can only course-correct *after* it has already overshot.

**What NAdam Changes**
NAdam incorporates Nesterov Accelerated Gradient (NAG), which flips the classical momentum logic:

| Logic Step | Classical Momentum (Adam) | Nesterov Momentum (NAdam) |
| --- | --- | --- |
| Order of operations | Look at slope → Adjust speed → Step | Leap based on past speed → Look at slope there → Correct course |
| Gradient evaluation | At current position $\theta_t$ | At anticipated (future) position |
| Reaction to change | After gradient is averaged over iterations | Immediately, via current $g_t$ |
| Braking power | Weak — relies on average slowly changing sign | Strong — current $g_t$ can instantly counter-act momentum |

By evaluating where the momentum *would* take the optimizer before committing to the full step, NAdam can begin correcting its trajectory before overshooting rather than after.

**The Mathematical "Cheat Code"**
Computing the gradient at a future position $\nabla J(\theta + \beta m)$ directly would require a second forward/backward pass — computationally expensive. Dozat's key insight was to **simulate the look-ahead algebraically** using only the current gradient $g_t$, with no extra computation.

In Adam, the update uses the bias-corrected first moment $\hat{m}_t$ directly. In NAdam, this is replaced by a **modified momentum** $\bar{m}_t$:
$$\bar{m}_t = \underbrace{\beta_1 \hat{m}_t}_{\text{Past momentum (decayed)}} + \underbrace{\frac{(1-\beta_1)\, g_t}{1-\beta_1^t}}_{\text{Instantaneous correction}}$$

- The **past momentum term** $\beta_1 \hat{m}_t$ carries the accumulated direction from all prior steps, decayed by $\beta_1$
- The **instantaneous correction term** directly injects the current gradient $g_t$ into this update with its full bias-corrected weight, reasoning: *"since $g_t$ will become part of my next step's momentum anyway, apply its influence right now"*

This effectively shifts the momentum calculation **one time-step forward** without any additional gradient evaluation — the look-ahead is achieved purely through algebraic rearrangement.

**Why This Is "Aggressive"**
Adam dilutes the current gradient's influence by averaging it into $m_t$ over several iterations before it meaningfully affects the update. NAdam gives the raw current gradient a **direct seat at the table** in the current update, bypassing the lag of the moving average.

**The downhill skier analogy:**
- **Adam:** High momentum from past steep gradients dominates. Even as the slope flattens at the minimum, $m_t$ is slow to reflect this. The optimizer flies past the bottom and oscillates.
- **NAdam:** The moment $g_t$ starts pointing in the opposite direction (indicating the minimum has been reached or passed), the instantaneous correction term acts as an **emergency brake** — countering the historical momentum before the optimizer travels too far in the wrong direction.

**Adam vs. NAdam at a Glance**

| Feature | Adam | NAdam |
| --- | --- | --- |
| **Momentum type** | Classical | Nesterov (look-ahead) |
| **Momentum lag** | Significant; reacts after averaging | Minimal; reacts instantly via $g_t$ |
| **Update rule** | Uses $\hat{m}_t$ directly | Uses modified $\bar{m}_t$ |
| **Path shape** | "Lazy" curves, may oscillate near minima | Sharp, responsive, hugs the minimum |
| **Convergence** | Standard, robust, gold standard | Often faster in complex landscapes |
| **Complexity** | Slightly simpler | Slightly more complex |

### NAdam vs Adam

| Aspect | Adam | NAdam |
| --- | --- | --- |
| **Stability** | Very stable | Stable, occasionally more sensitive |
| **Best for** | General purpose, default choice; reliable, well-tested optimizer | CNNs, vision tasks; when you have compute budget for potentially faster convergence; want to squeeze out extra performance from well-tuned models |

## Hyperparameters Comparison

| Optimizer  | Hyperparameters | Ones you actually tune |
| ---------- | --------------- | ---------------------- |
| Vanilla GD | η | η |
| Momentum   | η,γ | η |
| AdaGrad    | η,ϵ | η |
| RMSprop    | η,ρ,ϵ | η |
| AdaDelta   | ρ,ϵ | nothing (rarely even ρ) |
| Adam       | α,β₁,β₂,ϵ | α |