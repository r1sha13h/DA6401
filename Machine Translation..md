***

## Slide 1 — Title

**Many-to-Many RNNs for Machine Translation**
*From Encoder–Decoder to Attention*

***

## Slide 2 — Topics

**Already covered:**
- Many-to-one RNNs for sequence classification
- Backpropagation through time (BPTT)

**Now:**
- Many-to-many sequence modeling for machine translation
- Vanilla encoder–decoder RNN
- Why a fixed context vector is a bottleneck
- Attention as a learned soft alignment mechanism
- Why RNN-based translation still has limitations, even with attention

**Main pedagogical goal:**

classification → sequence generation → alignment → modern alternatives

***

## Slide 3 — Many-to-Many Setting in Machine Translation

Machine translation maps an input sequence in one language to an output sequence in another language.

\[ x_1, x_2, \ldots, x_{T_x} \longrightarrow y_1, y_2, \ldots, y_{T_y} \]

**Examples:**
- English → French
- Tamil → English
- Sentence lengths need not match: \(T_x \neq T_y\)

**Key difference from many-to-one:**
- In sentiment classification, one output label summarizes the whole sequence.
- In translation, the model must produce a *sequence* of outputs, one token at a time.

***

## Slide 4 — Probabilistic Formulation of Translation

The goal is to model the conditional probability of the target sentence given the source sentence:

\[ P(y_1, \ldots, y_{T_y} \mid x_1, \ldots, x_{T_x}) \]

Using the chain rule:

\[ P(y_1, \ldots, y_{T_y} \mid x) = \prod_{t=1}^{T_y} P(y_t \mid y_1, \ldots, y_{t-1}, x) \]

So at each decoder step \(t\), the model predicts:

\[ P(y_t \mid y_{<t}, x) \]

**Interpretation:**
- The next translated word depends on: the source sentence \(x\), and the target words already generated.
- Translation is **autoregressive generation**.

***

## Slide 5 — Vanilla Seq2Seq RNN: High-Level Idea

The classic RNN translation architecture has two parts:

**Encoder:**
- Reads the source sentence token by token
- Produces hidden states
- Final hidden state is used as a summary of the whole source

**Decoder:**
- Starts from this summary
- Generates target tokens one at a time
- Uses its own recurrent hidden state

**Core idea:**

source sentence → single vector → target sentence

This is elegant, but the single-vector summary becomes the central weakness.

***

## Slide 6 — Encoder Equations

Let the source tokens be \(x_1, \ldots, x_{T_x}\). Each token is embedded:

\[ e_t = E_x(x_t) \]

The encoder RNN updates:

\[ h_t = f_{\text{enc}}(e_t, h_{t-1}) \]

where:
- \(e_t\) = embedding of source token \(x_t\)
- \(h_t\) = encoder hidden state at time \(t\)
- \(f_{\text{enc}}\) can be a vanilla RNN, GRU, or LSTM cell

After the full source sentence is read, the final state is:

\[ c = h_{T_x} \]

**Interpretation:**
- \(c\) is the *context vector*
- It is supposed to summarize the entire source sentence

***

## Slide 7 — Decoder Equations

The decoder generates one target word at a time. Let the decoder hidden state at time \(t\) be \(s_t\).

**Initialization:**

\[ s_0 = \phi(c) \]

where \(c\) is the encoder context vector.

**At each decoder step:**

\[ s_t = f_{\text{dec}}(E_y(y_{t-1}),\, s_{t-1}) \]

Then logits are computed:

\[ o_t = W_o s_t + b_o \]

and the next-token distribution is:

\[ P(y_t \mid y_{<t}, x) = \text{softmax}(o_t) \]

**Special token:**
- Decoding begins from a `<BOS>` token
- Generation stops when `<EOS>` is produced

***

## Slide 8 — Why This Is Many-to-Many

This is a many-to-many architecture because:
- The input is a sequence: \(x_1, \ldots, x_{T_x}\)
- The output is also a sequence: \(y_1, \ldots, y_{T_y}\)

But unlike sequence labeling, translation is **not usually synchronous**:
- Input and output lengths differ
- Word order changes
- One source word may correspond to multiple target words
- Some target words may depend on long-range source context

So translation is a much harder many-to-many problem than tagging.

***

## Slide 9 — Diagram: Vanilla Encoder–Decoder

```
[x₁] [x₂] [x₃] ··· [xTx]              [y₁] [y₂] [y₃]
  ↓    ↓    ↓         ↓                   ↑    ↑    ↑
(h₁)→(h₂)→(h₃)→···→(hTx)→[context c]→(s₁)→(s₂)→(s₃)
```

> Everything the decoder knows about the source must pass through a single vector \(c\).

***

## Slide 10 — Training the Seq2Seq Model

Training usually uses **teacher forcing**. At decoder step \(t\), instead of feeding the model's own previous prediction, we feed the true previous target token:

\[ s_t = f_{\text{dec}}(E_y(y^{\text{true}}_{t-1}),\, s_{t-1}) \]

The loss is the sum of token-level cross-entropies:

\[ \mathcal{L} = -\sum_{t=1}^{T_y} \log P(y^{\text{true}}_t \mid y^{\text{true}}_{<t},\, x) \]

**Important point** — The gradient must flow:
- Backward through decoder time steps
- Through the context vector \(c\)
- Backward through encoder time steps

So training requires BPTT across a long computational graph.

***

## Slide 11 — Why the Fixed Context Vector Is a Bottleneck

In vanilla seq2seq, \(c = h_{T_x}\) must contain all information needed for the entire translation.

**That means one vector must encode:**
- Content of all source words
- Important long-range dependencies
- Ordering information
- Syntactic structure
- Semantic relations needed later in decoding

For short sentences this may be manageable. For long or complex sentences, it becomes unrealistic:

many words \(\not\to\) perfectly captured in one fixed-size vector

This is the **information bottleneck**.

***

## Slide 12 — Bottleneck (Continued)

Consider translating:
> *The book that the professor who visited the lab yesterday recommended was finally published.*

A single context vector must preserve:
- The main subject
- Subordinate clauses
- Who did what
- Tense and agreement information
- Content needed many decoding steps later

The decoder may need specific source details at different times:
- Early target tokens may depend on the source beginning
- Later target tokens may depend on a source word in the middle

A single summary vector does not allow selective retrieval of different source positions.

***

## Slide 13 — BPTT and the Bottleneck

It is useful to connect this to BPTT, but carefully.

In vanilla seq2seq, the loss at decoder step \(t\) influences encoder state \(h_k\) through a long path:

\[ y_t \to s_t \to s_{t-1} \to \cdots \to s_0 \to c = h_{T_x} \to h_{T_x - 1} \to \cdots \to h_k \]

So for early source words, gradients may have to traverse: many decoder steps, the context bottleneck, and many encoder steps.

**Consequences:**
- Vanishing gradients make learning long-range dependencies hard
- Hidden states are forced to preserve information over long spans
- The final encoder state is overloaded

So attention is motivated mainly by the bottleneck, and it also helps reduce this long-range optimization burden.

***

## Slide 14 — Key Idea of Attention

Instead of using only the final encoder state \(h_{T_x}\), let the decoder look at *all* encoder states: \(h_1, h_2, \ldots, h_{T_x}\).

At decoder step \(t\), compute a step-specific context vector:

\[ c_t = \sum_{j=1}^{T_x} \alpha_{t,j}\, h_j \]

where:
- \(\alpha_{t,j} \geq 0\)
- \(\sum_{j=1}^{T_x} \alpha_{t,j} = 1\)

**Interpretation:**
- \(\alpha_{t,j}\) tells how much decoder step \(t\) attends to source position \(j\)
- The decoder can focus on different source words at different times

***

## Slide 15 — Attention as Soft Alignment

In translation, there is often an implicit alignment between target and source words.

**Example:**

I eat apples → Je mange des pommes

When generating *mange*, the decoder should focus on *eat*. When generating *pommes*, it should focus on *apples*.

**Attention learns this softly:**
- Not one hard source position
- But a distribution over source positions: \(\alpha_{t,1},\, \alpha_{t,2},\, \ldots,\, \alpha_{t,T_x}\)

So attention is a **soft alignment mechanism**.

***

## Slide 16 — General Attention Computation

At decoder step \(t\), use the previous decoder state \(s_{t-1}\) and each encoder state \(h_j\) to compute a score:

\[ e_{t,j} = a(s_{t-1},\, h_j) \]

Then normalize using softmax:

\[ \alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_{k=1}^{T_x} \exp(e_{t,k})} \]

Finally compute the context vector:

\[ c_t = \sum_{j=1}^{T_x} \alpha_{t,j}\, h_j \]

Then the decoder state uses both the previous target token and this context:

\[ s_t = f_{\text{dec}}(E_y(y_{t-1}),\, s_{t-1},\, c_t) \]

Now the source representation is *dynamic* rather than fixed.

***

## Slide 17 — Typical Score Functions for Attention

The function \(a(s_{t-1}, h_j)\) can take different forms.

**Dot product:**

\[ e_{t,j} = s_{t-1}^\top h_j \]

**Bilinear / general form:**

\[ e_{t,j} = s_{t-1}^\top W_a h_j \]

**Additive attention (Bahdanau-style):**

\[ e_{t,j} = v_a^\top \tanh(W_s s_{t-1} + W_h h_j + b_a) \]

**Why additive attention became influential:**
- Flexible learned compatibility function
- Works even when encoder and decoder spaces differ

***

## Slide 18 — Decoder with Attention

Once \(c_t\) is computed, the decoder updates using source information specific to time step \(t\). A generic formulation is:

\[ s_t = f_{\text{dec}}(E_y(y_{t-1}),\, s_{t-1},\, c_t) \]

Then output logits may be formed using both the decoder state and the context:

\[ o_t = W_o[s_t;\, c_t] + b_o \]

and the next-token distribution is:

\[ P(y_t \mid y_{<t}, x) = \text{softmax}(o_t) \]

**Conceptually:**
- \(s_{t-1}\) says: *what have I generated so far?*
- \(c_t\) says: *which source information matters now?*

***

## Slide 19 — Diagram: Seq2Seq with Attention

```
[x₁] [x₂] [x₃] ··· [xTx]  [y₁]   [y₂] [y₃]
  ↓    ↓    ↓         ↓      ↑       ↑    ↑
(h₁)→(h₂)→(h₃)──────(hTx)  (s₁)──→(s₂)→(s₃)
  ╰────╰────╰──────────╯──────╯  ←── attention over all encoder states
```

> At each decoder step, the model can consult all source positions instead of relying on one fixed summary.

***

## Slide 20 — Why Attention Helps

Attention helps in several ways:

1. **Removes the fixed-length bottleneck** — The source is no longer compressed into only one vector.
2. **Gives step-specific access to the source** — Different target words can look at different source words.
3. **Improves gradient flow** — Decoder losses connect more directly to encoder states. The model need not preserve everything only in \(h_{T_x}\).
4. **Provides interpretability** — Attention weights often show approximate source-target alignments.

So attention is both an **architectural improvement** and an **optimization aid**.

***

## Slide 21 — How Attention Shortens the Information Path

Without attention, a decoder output may influence an early encoder state only through:

\[ y_t \to s_t \to \cdots \to s_0 \to h_{T_x} \to \cdots \to h_k \]

With attention, decoder step \(t\) uses \(h_k\) more directly through:

\[ h_k \to e_{t,k} \to \alpha_{t,k} \to c_t \to s_t \to y_t \]

**Important nuance:**
- The encoder is still recurrent.
- BPTT still exists.
- But the model no longer relies exclusively on the final encoder state.

This is why attention often works much better on long sentences.

***

## Slide 22 — Example of Attention During Translation

Suppose the source is: \(x = (\text{the, red, car})\)

and target is: \(y = (\text{la, voiture, rouge})\)

**At different decoder steps:**
- To produce *la*, attention may focus broadly
- To produce *voiture*, attention may peak on *car*
- To produce *rouge*, attention may peak on *red*

So the decoder is not asking: *"What was the whole sentence summary again?"*

It is asking: *"Which parts of the source sentence matter right now?"*

***

## Slide 23 — Teacher Forcing and Inference

**Training:**
- Use ground-truth previous token \(y^{\text{true}}_{t-1}\)
- Easier optimization

**Inference:**
- Previous token is the model's own prediction
- Errors can accumulate over time

At inference:

\[ \hat{y}_t = \arg\max_y\, P(y \mid \hat{y}_{<t}, x) \]

or use beam search.

**A practical issue:**
- During training, decoder sees true history
- During inference, decoder sees its own imperfect history

This affects both vanilla and attention-based RNN translation.

***

## Slide 24 — Does Attention Solve All Problems? No.

Attention is a major improvement, but RNN-based translation still has important limitations.

**Even with attention:**
- The encoder is still sequential
- The decoder is still sequential
- Long-range dependencies are still harder than in fully non-recurrent models
- Training remains difficult for very long sequences
- Inference remains slow because tokens are generated one by one

So attention improves seq2seq RNNs a lot, but does not remove the fundamental sequential nature of RNNs.

***

## Slide 25 — Transformer-Based MT

Transformers addressed several limitations.

**Key differences:**
- No recurrent hidden-state chain across time
- Self-attention connects tokens more directly
- Much better parallelization during training
- Easier modeling of long-range dependencies

**Very roughly:**

Vanilla RNN MT → Attention-based RNN MT → Transformer MT

So attention started as an improvement *inside* RNN seq2seq, and later became the central mechanism of a new architecture family.

***

## Slide 26 — Summary

**Vanilla encoder–decoder RNN:**
- Encodes source into one vector \(c\)
- Decoder generates target from \(c\)
- Suffers from a fixed-context bottleneck

**Attention-based seq2seq:**
- Decoder attends to all encoder states
- Context becomes dynamic: \(c_t\)
- Improves alignment, learning, and long-sentence handling

**But even with attention:**
- Recurrence remains sequential
- Training and inference remain slower
- Long-range modeling is still imperfect

This motivates the transition to Transformers.

***

## Slide 27 — Comparison Table

| | **Vanilla Seq2Seq RNN** | **Seq2Seq RNN + Attention** |
|---|---|---|
| Source representation | Single context vector \(c\) | All encoder states \(\{h_j\}\) |
| Decoder access to source | Indirect, fixed | Dynamic, step-specific |
| Long sentence handling | Weak | Better |
| Alignment behavior | Implicit only | Soft alignment via \(\alpha_{t,j}\) |
| Optimization | Hard over long paths | Improved, but still recurrent |
| Parallelism | Poor | Poor to moderate |
| Still limited by recurrence? | Yes | Yes |