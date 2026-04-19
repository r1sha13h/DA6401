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

### RMSprop Algorithm
1. **Initialize**: parameters $\theta_0$, base learning rate $\eta$, decay $\beta$ (≈0.9), moving average $E[g^2]_0 = 0$, small constant $\epsilon = 10^{-8}$.
2. **Iterate for each step $t$**:
   1. Compute gradient $g_t = \nabla_{\theta} J(\theta_t)$.
   2. Update EMA: $E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta)(g_t \odot g_t)$.
   3. Parameter update: $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \odot g_t$.

**AdaDelta Algorithm:**
1. Initialize: $\theta_0$, decay rate $\rho$ (typically 0.9 or 0.95)
2. Initialize: $E[g^2]_0 = 0$, $E[\Delta \theta^2]_0 = 0$, $\epsilon = 10^{-6}$
3. For $t = 1$ to $T$:
   - Compute gradient: $g_t = \nabla_\theta J(\theta_t)$
   - Accumulate gradient: $E[g^2]_t = \rho E[g^2]_{t-1} + (1-\rho) g_t \odot g_t$
   - Compute update: $\Delta \theta_t = -\frac{\sqrt{E[\Delta \theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} \odot g_t$
   - Accumulate updates: $E[\Delta \theta^2]_t = \rho E[\Delta \theta^2]_{t-1} + (1-\rho) \Delta \theta_t \odot \Delta \theta_t$
   - Update parameters: $\theta_{t+1} = \theta_t + \Delta \theta_t$

**Adam Algorithm:**
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

**Nadam Algorithm:**
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

### Important Points
- Identity shortcuts add **zero extra parameters**
- The $+1$ in the gradient equation is the entire mathematical reason skip connections prevent vanishing gradients
- **Batch Normalization** is used — not LRN like in AlexNet
- ResNet's use of Global Average Pooling reduces the classifier head from $\approx 119$ million parameters (as in VGG) to just $512$ thousand, and removes the fixed input size constraint imposed by fully connected layers.

## Architecture Comparison

| Model | Depth | Params (M) | Top-5 Error (%) |
|-------|-------|------------|-----------------|
| LeNet-5 | $7$ | $60$ thousand | - |
| AlexNet | $8$ | $60$ million | $\approx 16.4\%$ |
| VGG-16 | $16$ | $138$ million | $\approx 7.3\%$ |
| GoogLeNet | $22$ | $5$ million | $\approx 6.7\%$ |
| ResNet-50 | $50$ | $25$ million | $\approx 5.3\%$ |
| ResNet-152 | $152$ | $60$ million | $3.57\%$ |

#### Algorithm

For a mini-batch $B = \{x_1, \dots, x_m\}$:

**Step 1 — Batch Mean:**
$$\mu_B = \frac{1}{m} \sum_{i} x_i$$

**Step 2 — Batch Variance:**
$$\sigma_B^2 = \frac{1}{m} \sum_{i} (x_i - \mu_B)^2$$

**Step 3 — Normalize:**
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

**Step 4 — Scale & Shift:**
$$y_i = \gamma \hat{x}_i + \beta$$

- $\gamma, \beta$ — learnable parameters, one per feature
- $\epsilon$ — small constant for numerical stability (prevents division by zero)

$w \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{in} + n_{out}}},\ +\sqrt{\frac{6}{n_{in} + n_{out}}}\right]$

$w \sim \mathcal{N}\left(0,\ \sqrt{\frac{2}{n_{in}}}\right)$

## Key Principle

| Technique | Mechanism | Primary Role |
|---|---|---|
| **Dropout** | Random neuron silencing | Prevents co-adaptation → generalization |
| **Data Augmentation** | Label-preserving input transforms | Increases effective data diversity |
| **Weight Decay (L2)** | Penalizes large weights in loss | Prevents over-reliance on specific features |
| **Weight Initialization** | Sets starting weight scale correctly | Ensures stable training from step 1 |

## Hierarchy of Tasks
| Task | Output | Key Complexity |
|---|---|---|
| **Classification** | Single label — *"there's a cat"* | Simplest — one answer per image |
| **Localisation** | Class label + one bounding box | One object only |
| **Detection** | All objects + bounding boxes | Variable number of objects |
| **Segmentation** | Class label for every pixel | Finest granularity |

5. Filter → NMS → top ~300 proposals

| | Selective Search | RPN |
|---|---|---|
| Runs on | CPU | GPU |
| Time | ~1,500ms | ~10ms |
| Learned? | ✗ | ✓ |
| Shares CNN features? | ✗ | ✓ |
| Output proposals | ~2,000 | ~300 |

## R-CNN Family — Full Comparison
| | R-CNN | Fast R-CNN | Faster R-CNN |
|---|---|---|---|
| **Proposal method** | Selective Search | Selective Search | **RPN (learned)** |
| **Proposal count** | ~2,000 | ~2,000 | **~300** |
| **CNN runs per image** | 2,000 | **1** | **1** |
| **GPU for proposals?** | ✗ | ✗ | **✓** |
| **End-to-end trainable?** | ✗ | Partially | **✓ fully** |
| **Training time** | 84 hours | Much faster | Faster |
| **Inference speed** | 47 sec | ~2 sec | **~0.2 sec** |

### Performance (at Release)
| Model | FPS | mAP (PASCAL VOC) | Trade-off |
|---|---|---|---|
| **SSD300** | **59 fps** | **74.3%** | Best speed/accuracy balance |
| **YOLOv1** | 45 fps | Lower | Faster but less accurate |
| **Faster R-CNN** | ~5 fps | Higher | More accurate but slower |

## Object Detection Methods: Comparison

| Method | Year | mAP | Speed | Stage | Key Idea |
|---|---|---|---|---|---|
| **R-CNN** | 2014 | 58.5 | 47s/img | 2-stage | Region proposals + CNN |
| **Fast R-CNN** | 2015 | 70.0 | 2s/img | 2-stage | RoI Pooling |
| **Faster R-CNN** | 2015 | 73.2 | 200ms/img | 2-stage | RPN (end-to-end) |
| **YOLO v1** | 2016 | 63.4 | 22ms/img | 1-stage | Grid + single forward pass |
| **SSD300** | 2016 | 74.3 | 17ms/img | 1-stage | Multi-scale anchors |
| **YOLOv3** | 2018 | 82.1 | 29ms/img | 1-stage | Multi-scale + Darknet53 |