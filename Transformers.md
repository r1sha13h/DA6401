## Slide 1 — Title Slide

**From RNN Attention to Transformers**
*Self-Attention, Positional Encoding, and Modern Machine Translation*

***

## Slide 2 — Where We Are Coming From

**In the previous lecture:**
- Vanilla seq2seq RNN translation used one fixed context vector
- Attention improved this by giving the decoder access to all encoder states
- But even with attention, the model was still recurrent

**Remaining problems:**
- Sequential computation
- Slow training
- Slow inference
- Difficulty with very long-range dependencies

**Key question:** If attention is the key idea, do we really need recurrence at all?

***

## Slide 3 — A New Principle: Every Token Can Look at Every Other Token

Given token representations \(x_1, x_2, \ldots, x_T\), the representation for token \(i\) is:

$$
z_i = \sum_{j=1}^{T} \alpha_{i,j} x_j
$$

Where:
- \(\alpha_{i,j}\) says how much token \(i\) should attend to token \(j\)
- The weights depend on **content**
- Each token gets its own weighted summary

This is the core idea of **self-attention**.

***

## Slide 4 — From Attention in Seq2Seq to Self-Attention

**In RNN translation with attention:**
- Decoder state queried encoder states
- This was **cross-attention** in spirit

**In self-attention:**
- Each token queries the other tokens in the *same* sequence
- Source tokens attend to source tokens
- Target tokens attend to earlier target tokens

**Progression:**
> encoder–decoder attention → self-attention within a sequence

This lets the model build context-rich token representations without recurrence.

***

## Slide 5 — Queries, Keys, and Values

Self-attention uses three learned projections for each token \(x_i\):

$$
q_i = W_Q x_i, \quad k_i = W_K x_i, \quad v_i = W_V x_i
$$

**Interpretation:**
- **Query:** what this token is looking for
- **Key:** what this token offers for matching
- **Value:** the information this token contributes if selected

Similarity between token \(i\) and token \(j\) is measured by:

$$
\text{sim}(i,j) = q_i^\top k_j
$$

Token \(i\) then forms a weighted combination of the values \(v_j\).

***

## Slide 6 — Self-Attention Equations

**Step 1 — Compute scores:**
$$
e_{i,j} = q_i^\top k_j
$$

**Step 2 — Scale and normalize:**
$$
\alpha_{i,j} = \frac{\exp\!\left(\dfrac{q_i^\top k_j}{\sqrt{d_k}}\right)}{\sum_{m=1}^{T} \exp\!\left(\dfrac{q_i^\top k_m}{\sqrt{d_k}}\right)}
$$

**Step 3 — Compute output:**
$$
z_i = \sum_{j=1}^{T} \alpha_{i,j} v_j
$$

**Important points:**
- Every token can aggregate information from all other tokens
- Relevant tokens get larger weights

***

## Slide 7 — Matrix Form of Self-Attention

Collect all token representations row-wise into \(X \in \mathbb{R}^{T \times d}\). Then:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

Self-attention in matrix form:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

**Properties:**
- All pairwise token interactions are computed together
- Highly parallelizable on modern hardware
- No recurrent chain across time

***

## Slide 8 — Intuition: Why Self-Attention Helps

Consider: *"The animal didn't cross the street because it was tired."*

To understand *it*, the model needs to relate it to *animal*.

| Model | Path |
|-------|------|
| **RNN** | Information travels through many recurrent steps |
| **Self-attention** | The token *it* can directly attend to *animal* |

The path between distant tokens becomes much shorter.
> RNN path length **grows with distance** vs. self-attention path length is **direct**

***

## Slide 9 — Order of Tokens

A pure attention mechanism does **not** inherently know token order. Without extra information, these sequences would look the same:

> (dog, bites, man) and (man, bites, dog)

**Solution: Positional Encoding** — add a position-dependent vector to each token:

$$
x_i^{(\text{input})} = e_i + p_i
$$

Where:
- \(e_i\) is the token embedding
- \(p_i\) is the positional encoding

***

## Slide 10 — Sinusoidal Positional Encoding

$$
PE(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)
$$
$$
PE(\text{pos}, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)
$$

**Why this is useful:**
- Gives each position a distinct pattern
- Lets the model infer relative positions from combinations of sine and cosine values
- Does not require recurrence

In practice, modern models may use learned or relative positional representations, but the core idea is:
> **Order must be injected explicitly**

***

## Slide 11 — What Is a Transformer Encoder Block?

A Transformer encoder block contains:
1. Multi-head self-attention
2. Add & LayerNorm
3. Position-wise feedforward network
4. Add & LayerNorm

**Symbolic flow:**
> \(X \rightarrow \text{Self-Attention} \rightarrow \text{Add\&Norm} \rightarrow \text{FFN} \rightarrow \text{Add\&Norm}\)

**Role of the encoder:**
- Build contextualized representations of source tokens
- Each source token becomes aware of relevant other source tokens

***

## Slide 12 — What Is a Transformer Decoder Block?

A Transformer decoder block has three main parts:
1. **Masked self-attention** over previously generated target tokens
2. **Cross-attention** to encoder outputs
3. **Feedforward network**

**Why masked self-attention?**
When predicting \(y_t\), the model must not look at future tokens. Attention is masked to allow only \(y_1, \ldots, y_{t-1}\) when predicting \(y_t\).

**Cross-attention:** Target-side states query the encoder outputs.

***

## Slide 13 — Diagram of the Transformer for Translation

```
Source embeddings          Target embeddings
   + positions                + positions
       ↓                           ↓
  Encoder stack           Masked self-attention
 (self-attention               ↓
    + FFN)              Cross-attention to
       ↓                  encoder outputs ←──
  Encoder outputs               ↓
                          FFN + output layer
```

The decoder uses:
- Target-side masked self-attention
- Encoder–decoder cross-attention

***

## Slide 14 — Why Multi-Head Attention?

A single attention mechanism produces one weighted view of the sequence. Multi-head attention runs several patterns in parallel:

$$
\text{head}_r = \text{Attention}(Q_r, K_r, V_r)
$$
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)W_O
$$

**Intuition — each head can specialize:**
- One head may focus on **local structure**
- Another on **long-range dependencies**
- Another on **syntactic relations**

Multi-head attention gives **multiple learned views of relevance**.

***

## Slide 15 — Comparison: RNN Attention vs. Transformer

| Aspect | RNN + Attention | Transformer |
|---|---|---|
| Sequence processing | Sequential | Parallel across positions |
| Main mechanism | Recurrence + attention | Self-attention + cross-attention |
| Long-range interaction | Improved but indirect | More direct |
| Order information | Comes from recurrence | Added explicitly via positional encoding |
| Training speed | Slower | Faster on modern hardware |
| Scalability | Limited | Stronger |

***

## Slide 16 — Summary

- Attention suggested a deeper principle: **keep representations available and retrieve what is relevant**
- Transformers apply this principle directly within sequences using self-attention
- Queries, keys, and values define content-based interaction
- Positional encoding supplies order without recurrence
- The decoder uses masked self-attention and cross-attention
- This removes the main limitations of RNN-based translation

> **RNN memory → attention retrieval → Transformer**

## Slide 1 — Title Slide

**Cross-Attention in Transformers**
*Mechanism, Mathematics, and Applications*
Introduction to Deep Learning | April 24, 2026

***

## Slide 2 — Outline

1. Motivation & Background
2. Mathematical Formulation
3. Architecture: Cross-Attention
4. Masked Self-Attention in the Decoder
5. Intuition & Visualisation
6. Computational Considerations
7. Variants & Extensions
8. Summary

***

## Slide 3 — What Problem Does Cross-Attention Solve?

**The Encoder-Decoder Challenge:**
- Many tasks require mapping one sequence to another (translation, summarisation, speech recognition)
- Encoder produces a rich contextual representation of the source
- Decoder must selectively attend to **relevant parts** of the source while generating the target
- A fixed-length bottleneck vector loses information for long sequences

> **Key insight:** Instead of compressing the source to a single vector, let the decoder *query* the encoder at every step.

**Diagram:** Source Seq. → Encoder → fixed vector ❓ (bottleneck!) → Decoder → Target Seq. [labelled "Old approach"]

***

## Slide 4 — Self-Attention vs. Cross-Attention

**Diagram (left): Self-Attention**
- Tokens: "He", "ate", "pizza" — Q, K, V all from same sequence → Attention → Output

**Diagram (right): Cross-Attention**
- Source (Encoder): "Il", "mangia", "pizza" → provides K, V
- Target (Decoder): "He", "ate" → provides Q
- Both feed into Cross-Attention block

| | Self-Attention | Cross-Attention |
|---|---|---|
| Q, K, V source | Same sequence | Q from decoder; K, V from encoder output |
| Used in | Encoder and decoder (masked) | Bridges two different sequences |

***

## Slide 5 — Scaled Dot-Product Attention (Recap)

**General Attention Formula (Vaswani et al., 2017):**

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

**Tensor shapes:**

$$
\begin{aligned}
Q &\in \mathbb{R}^{T_q \times d_k} \quad \text{(query matrix)} \\
K &\in \mathbb{R}^{T_k \times d_k} \quad \text{(key matrix)} \\
V &\in \mathbb{R}^{T_k \times d_v} \quad \text{(value matrix)} \\
\text{Output} &\in \mathbb{R}^{T_q \times d_v}
\end{aligned}
$$

**Why scale by \(1/\sqrt{d_k}\)?**
- Prevents dot products from growing very large
- Keeps softmax in a well-behaved gradient region

**Scores Matrix:** 

$$
S = \dfrac{QK^\top}{\sqrt{d_k}} \in \mathbb{R}^{T_q \times T_k}
$$

Where \(S_{ij}\) is the relevance of source token \(j\) to query token \(i\).

**Attention Weights:** 

$$
A = \text{softmax}_\text{row}(S) \in \mathbb{R}^{T_q \times T_k}
$$

Each row sums to 1.

***

## Slide 6 — Cross-Attention: Precise Definition

Let \(X \in \mathbb{R}^{T_x \times d_\text{model}}\) be the **encoder output** and \(Y \in \mathbb{R}^{T_y \times d_\text{model}}\) be the **decoder hidden state**.

**Step 1 — Linear projections:**

$$
Q = YW^Q, \quad K = XW^K, \quad V = XW^V
$$

where \(W^Q, W^K \in \mathbb{R}^{d_\text{model} \times d_k}\), \(W^V \in \mathbb{R}^{d_\text{model} \times d_v}\)

**Step 2 — Compute cross-attention:**

$$
\text{CrossAttn}(X,Y) = \text{softmax}\!\left(\frac{(YW^Q)(XW^K)^\top}{\sqrt{d_k}}\right)(XW^V)
$$

**Key Asymmetry:**
- Query length is \(T_y\) (target); key/value length is \(T_x\) (source)
- \(T_x \neq T_y\) in general — cross-attention handles variable-length pairs naturally

***

## Slide 7 — Step-by-Step Computation

**1. Project to Q, K, V** — for each decoder position \(i\), encoder position \(j\):

$$
q_i = y_i W^Q, \quad k_j = x_j W^K, \quad v_j = x_j W^V
$$

**2. Compute attention score:**

$$
e_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}
$$

**3. Normalise over source positions:**

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{j'=1}^{T_x} \exp(e_{ij'})}
$$

**4. Weighted sum of values:**

$$
z_i = \sum_{j=1}^{T_x} \alpha_{ij} v_j
$$

**Diagram (pipeline):** Project Q, K, V → Dot product \(QK^\top\) \([\mathbb{R}^{T_y \times T_x}]\) → Scale by \(1/\sqrt{d_k}\) → Softmax (over source, row-wise) → Weighted sum with V \([\mathbb{R}^{T_y \times d_v}]\)

***

## Slide 8 — Multi-Head Cross-Attention

**Motivation:** Different heads attend to different aspects of the source (syntax, semantics, position).

For each head \(h = 1, \ldots, H\):

$$
\text{head}_h = \text{Attention}(YW_h^Q,\; XW_h^K,\; XW_h^V)
$$

$$
\text{MultiHead}(X,Y) = [\text{head}_1 \| \cdots \| \text{head}_H] W^O, \quad W^O \in \mathbb{R}^{Hd_v \times d_\text{model}}
$$

| Typical Hyperparameters (base model) | |
|---|---|
| \(d_\text{model}\) | 512 |
| \(H\) (heads) | 8 |
| \(d_k = d_v\) | 64 |
| \(H \times d_k = 512 = d_\text{model}\) | ✓ |

**Parameter Count:**
- Per head: \(3 \times d_\text{model} \times d_k\)
- Total Q, K, V: \(3 \times d_\text{model}^2\)
- Output projection: \(d_\text{model}^2\)
- **Grand total: \(4d_\text{model}^2\)**

***

## Slide 9 — The Original Transformer Architecture

**Diagram: Encoder (left) → Decoder (right)**

**Encoder stack (×N):**
Input + Pos. Emb. → Multi-Head Self-Attn → Add & Norm → Feed-Forward → Add & Norm → **Encoder Output X**

**Decoder stack (×N):**
Output + Pos. Emb. → Masked Self-Attn → Add & Norm → **Cross-Attn** (receives K, V from Encoder Output X; Q from decoder) → Add & Norm → Feed-Forward → Add & Norm → Linear + Softmax

***

## Slide 10 — Inside a Decoder Layer

**Detailed diagram of one decoder block \(\ell\):**

Decoder Input \(Y^{(\ell-1)}\)
→ Masked Multi-Head Self-Attn + residual
→ Add & Norm
→ **Multi-Head Cross-Attention** (Q from decoder; K, V from Encoder Output X) + residual
→ Add & Norm
→ Position-wise FFN + residual
→ Add & Norm
→ Decoder Output \(Y^{(\ell)}\)

***

## Slide 11 — The Residual + LayerNorm Wrapper

Each sublayer (including cross-attention) is wrapped identically:

$$
Y^\text{out} = \text{LayerNorm}(Y^\text{in} + \text{MultiHeadCrossAttn}(X, Y^\text{in}))
$$

**Residual Connection:**
- Allows gradients to flow directly through layers
- Enables training of very deep networks
- Output = input + sublayer transformation

**Layer Normalisation:**
$$
\text{LN}(\mathbf{h}) = \frac{\mathbf{h} - \mu}{\sigma + \varepsilon}\,\gamma + \beta
$$
Normalises over the *feature* dimension per token. Stabilises training.

**Pre-LN vs. Post-LN:**
- **Post-LN** (original paper): LN after residual addition — can be unstable early in training
- **Pre-LN** (GPT-2 and later): LN *before* sublayer — more stable, now standard

***

## Slide 12 — Why the Decoder Needs a Mask

At training time the full target sequence is fed to the decoder all at once (for parallelism). But token \(i\) must **only** depend on tokens \(1, \ldots, i-1\) — not on future tokens.

**Without a mask:**
Token 3 could directly attend to token 5, "seeing the answer" before predicting it. The model would learn a trivial copy task rather than genuine language modelling.

**Solution: Causal (look-ahead) mask** — set all attention scores to the future to \(-\infty\) before the softmax; after softmax those weights become exactly 0.

**Diagram:** "Attention from 'the'" in sentence "He ate the pizza":
- "He" ✓ ok, "ate" ✓ ok, "the" ✓ self, "pizza" ✗ blocked
- *Can see past; cannot see future*

***

## Slide 13 — The Causal Mask: Construction & Effect

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

where the causal mask \(M \in \mathbb{R}^{T \times T}\) is:

$$
M_{ij} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}
$$

**Effect:** \(\exp(-\infty) = 0\), so after softmax future positions receive **exactly zero weight**.

**In practice:** \(-\infty\) is implemented as \(-10^9\) in PyTorch to avoid `NaN` from true infinity.

**Causal mask M for \(T = 4\) (visual lower-triangular matrix):**

| | j=1 | j=2 | j=3 | j=4 |
|---|---|---|---|---|
| i=1 | 0 | 0 | 0 | 0 |
| i=2 | 0 | 0 | 0 | \(-\infty\) |
| i=3 | 0 | 0 | \(-\infty\) | \(-\infty\) |
| i=4 | 0 | \(-\infty\) | \(-\infty\) | \(-\infty\) |

*(Green = allowed, Red = blocked)*

***

## Slide 14 — Three Attention Types: Side-by-Side

| | Q from | K, V from | Mask |
|---|---|---|---|
| **Encoder Self-Attn** | Encoder tokens | Encoder tokens | None — sees full source freely |
| **Decoder Masked Self-Attn** | Decoder tokens | Decoder tokens | **Causal** — token \(i\) sees only positions \(1\ldots i\) |
| **Decoder Cross-Attn** | Decoder tokens | Encoder tokens | Padding mask only (optional) — sees all source positions |

**Common Misconception:** "Cross-attention is masked" — it is **not** causally masked. The causal mask lives exclusively in the *decoder self-attention* sublayer.

**Why cross-attention needs no causal mask:** The encoder output X is a complete, fixed sequence. There are no "future" encoder tokens to hide — the decoder may look at all of them simultaneously.

***

## Slide 15 — What Do the Attention Weights Mean?

Recall: weight \(\alpha_{ij}\) captures how much **decoder token \(i\)** attends to **encoder token \(j\)**.

**Machine Translation Example (English → French):**
- Generating *"zone"* → high attention on *"area"*
- Generating *économique* → high attention on *"economic"*
- Adjective–noun reordering handled automatically

**Diagram:** Attention heatmap — Source (English): "The", "econ.", "policy", "area" on x-axis; Target (French): "La", "zone", "de", "polit.", "écon." on y-axis. High-intensity cells reveal word-level alignments.

**Interpretability:** Attention maps provide a window into what the model focuses on — valuable for debugging and analysis.

***

## Slide 16 — Query–Key–Value Analogy

**Database Analogy:**
> **Query** (What am I looking for?) → match → **Keys** (Index of what is stored) → retrieve → **Values** (Actual stored content)

**In Cross-Attention:**

| Q from decoder | K from encoder | V from encoder |
|---|---|---|
| *What to generate?* | *Source structure* | *Source content* |

**Intuition:** The decoder asks: *"Given what I want to generate (Q), which parts of the source (K) are most relevant, and what information do I retrieve from them (V)?"*

***

## Slide 17 — KV-Caching at Inference

**Why it matters:** During autoregressive decoding, the encoder output X is **fixed**. We only need \(K = XW^K\) and \(V = XW^V\) **once per source**.

**Pipeline diagram:**
1. Encode source once
2. Cache K, V in all layers ← *computed once, stored in memory*
3. Decode token by token
4. Reuse cached K, V → loop back for next token

**Speedup:** Reduces per-step cross-attention cost from \(\mathcal{O}(T_x d^2)\) to \(\mathcal{O}(T_x d)\).

***

## Slide 18 — Cross-Attention Beyond NLP

| Domain | Source (K, V from) | Query (Q from) |
|---|---|---|
| Machine Translation | Source sentence | Partial translation |
| Speech Recognition | Audio encoder | Text decoder |
| Image Captioning | Vision encoder | Caption decoder |
| Text-to-Image | Text encoder | Diffusion UNet |
| Multimodal LLMs | Image/video tokens | Language tokens |
| Perceiver IO | High-dim inputs | Latent array |
| Robotics (RT-2) | Language goal | Action decoder |

**General Principle:** Whenever you need to **condition one modality or sequence on another**, cross-attention provides a flexible, differentiable alignment mechanism.

***

## Slide 19 — Summary

**Core ideas:**
- Cross-attention lets the **decoder** selectively retrieve information from the **encoder**
- Q comes from the decoder; K, V come from the encoder output
- Generalises to any source → target alignment problem
- Handles variable-length pairs with no additional mechanism

**Key properties:**
- \(\mathcal{O}(T_y T_x d)\) time complexity
- Fully parallel (no recurrence)
- KV-caching enables efficient autoregressive decoding
- Multi-head captures diverse alignment patterns

**Applications:** MT, ASR, image captioning, text-to-image, multimodal LLMs, robotics, …