M O D U L E         5




NLP &
Sequence Models
     L1: Seq2Seq & Embeddings       L2: RNNs & BPTT   L3: LSTMs & GRUs   L4: Seq2Seq & Attention




Introduction to Deep Learning

4 Lectures · 50 min each · Week 5
LEC TUR E 1    ·   MODULE       5




Seq2Seq Tasks & Word Embeddings


•
•
•
    NLP tasks and why sequences differ from fixed-size inputs
    From one-hot encoding to dense vector representations
    Word2Vec Skip-gram: objective and full derivation
                                                                1
•   Negative sampling (SGNS) and the noise distribution
•   Vector space geometry and semantic analogies
                                                                                                                           L1 · NLP & Embeddings
NLP: Task Landscape

       Sentiment                 Machine                                     Language                 Question
                                                           NER                                                             Summarisation
        Analysis                Translation                                  Modelling               Answering

Seq → Label                Seq → Seq              Seq → Seq            Seq → next token         (Q,Doc) → Span          Long → Short seq




"Great film!" → Positive   "Hello" → "Hola"       "Apple Inc." → ORG   "The cat sat" → "on"     "Who?" + doc → answer   article → headline




                                                                       Why Standard Architectures Fail
 Variable-length sequences                                               •      MLP: fixed input/output size; destroys sequential
                                                                                structure
   •      Input and/or output length is not fixed — unlike images
                                                                         •      CNN: captures local patterns but no notion of long-range
          (H×W×C)
                                                                                order
   •      Tokens are discrete symbols — not natural numbers
                                                                         •      Both require a fundamentally different inductive bias:
   •      Context matters: 'bank' near 'river' ≠ 'bank' near 'money'
                                                                                recurrence
   •      Order matters: 'dog bites man' ≠ 'man bites dog'
                                                                         •      We need to process tokens one at a time, maintaining
                                                                                state
                                                                                                      L1 · NLP & Embeddings
From One-Hot Vectors to Dense Embeddings

One-Hot Encoding
                                                             Embedding lookup:

                                                               0
 •    Vocabulary V = {"the","cat","sat","on","mat"}, |V|=5
 •    "cat" → [0, 1, 0, 0, 0]
 •    Dimension = |V| — up to 50,000 in practice               0
 •    Dot product of any two one-hot vectors = 0 (all

 •
      orthogonal)
      No similarity: cat·dog = 0, same as cat·airplane = 0
                                                               1
                                                                   ← 'sat'
                                                                                     E
                                                               0                                         d-dim
                                                                                                         vector

Dense Embeddings                                               0
 •    Map each word index to a low-dimensional real vector
 •    Embeddings are learned — similar words get similar                     |V|×d embedding matrix
      vectors
 •    One-hot × E = lookup of row w in E
 •    Parameters: |V|×d (e.g. 50K × 300 = 15M)
                                                                                                                       L1 · NLP & Embeddings
Word2Vec: Skip-gram Architecture

                                                                 Skip-gram: one centre word → C context words
Core Idea (Mikolov et al., 2013)
 •     Given a centre word, predict surrounding context words
 •     Hypothesis: words with similar contexts have similar                            wₜ (centre word)
       meanings
 •     Context window size k: words within ±k positions
 •     Two embedding matrices: W (input) and W'
       (output/context)                                                                Lookup vwₜ in W




                                                                                       Hidden: v wₜ ∈ ℝᵈ


Why two embedding matrices?
 •     If W = W': self-score = ||v_w||² — always highest for
                                                                      wₜ₋₂      wₜ₋₁         wₜ            wₜ₊₁      wₜ₊₂
       any word
 •     Asymmetry is necessary to break trivial self-prediction
                                                                                        context words (window k=2)
After training
 •     W is kept as word embeddings; W' is discarded
 •     Some implementations average W and W' (double
       embeddings)
                                                                                                                               L1 · NLP & Embeddings
 Skip-gram: Probability Model & Derivation

Deriving P(context | centre):

 Step 1 — Similarity score: dot product of output and input vectors



 vᶜ ∈ ℝᵈ from input matrix W; ṽₒ ∈ ℝᵈ from output matrix W'.


 Step 2 — Softmax over all V words gives probability distribution



 Denominator sums over ALL words in vocabulary — expensive for large |V|!


 Step 3 — Negative log-likelihood loss for one (centre, context) pair



 Minimising J pushes the correct pair score up and all other pairs down.


 Step 4 — Gradient w.r.t. centre vector vᶜ



 Gradient = −(true context vector) + (expected context vector under current model). Push vᶜ toward ṽₒ, away from all others.
                                                                                                                   L1 · NLP & Embeddings
Negative Sampling (SGNS)

                                                                  Noise distribution:
 The Problem with Full Softmax

  •     Σ_{w∈V} exp(ṽwᵀ vᶜ) requires |V| dot products per step
  •     With |V|=50K and d=300: millions of multiplications per
        gradient step
  •     Solution: replace the 50K-way classifier with a binary
        one                                                         Gradient (only k+1 dot products)


 SGNS: Binary Classifier
                                                                     •     Push vᶜ toward true context ṽₒ
  •     Is (centre, context) pair real or noise?                     •     Push vᶜ away from k noise words ṽᵢ
  •     Sample k negatives from P_n(w) ∝ freq(w)^(3/4)
  •     k = 5–20 in practice — thousands of times cheaper
  •     Both W and W' are still fully learned via backprop

SGNS Objective:

                                                                                           Full Softmax         SGNS

                                                                   Dot products            |V| ≈ 50K            k+1 ≤ 21
                                                                   W' rows updated         All |V|              k+1 only
                                                                   Speedup                 —                    ~2,500×
                                                                                                      L1 · NLP & Embeddings
Word Vectors: Geometric Properties

                                                                 Vector space (2D projection):
 The Analogy Test


   •     Vector arithmetic captures semantic relationships:
   •     Emerged purely from co-occurrence statistics

                                                                                    king         queen




                                                                                      royalty    gender

Cosine Similarity:

                                                                                      man        woman




 Known Limitations
   •     Polysemy: 'bank' has one vector for multiple meanings
   •     Static: same vector regardless of context
   •     Fix: contextualised embeddings (BERT, GPT) — Module 6
LEC TUR E 2    ·   MODULE 5




RNNs & Backprop Through Time


•
•
•
    RNN architecture: the recurrence equation
    Forward pass — all 4 equations
    BPTT: full gradient derivation
                                                             2
•   Vanishing & exploding gradients: mathematical analysis
•   Gradient clipping
                                                                                                                   L2 · RNNs & BPTT
 Recurrent Neural Network: Core Architecture

The Recurrence Equations:
                                                                         ŷ₁                   ŷ₂        ŷ₃                   ŷ₄

                                                                RNN unrolled for T=4 steps:


                                                                h₀        h₁              h₂            h₃                    h₄

                                                                                      Wₕₕ (shared)


 Parameter count
   •     Wₕₕ: H×H Wₓₕ: H×d bₕ: H Wₕᵧ: C×H bᵧ: C                                                                               x₄
                                                                              x₁                   x₂        x₃
                                                                                                              x₃
   •     Total: H(H+d+1) + C(H+1) — independent of sequence
         length T!
   •     Same matrices applied at every timestep (weight
         sharing)

 Initialisation
   •     h₀ = 0 (zero vector, or learned parameter)
   •     First hidden state has no history — computed from x₁
         only
                                                                                       L2 · RNNs & BPTT
 RNN Forward Pass: Full Equations

Complete forward pass for x₁, x₂, …, xT:

 Pre-activation — linear combination of previous state and current input



 zₜ ∈ ℝᴴ. Two matrix-vector products: H×H and H×d. Additive bias bₕ.


 Hidden state — non-linear activation applied element-wise



 tanh maps ℝ → (−1,1). Saturates for large |z| — a source of vanishing gradients.


 Output logits — only computed at steps where prediction is needed



 C = vocabulary size for language modelling. Apply softmax to get ŷₜ = softmax(oₜ).


 Cross-entropy loss at step t



 yₜ is the ground-truth index. Total loss: L = (1/T) Σₜ Lₜ (average over timesteps).
                                                                            L2 · RNNs & BPTT
 Backpropagation Through Time (BPTT): Derivation

Goal: compute ∂L/∂Wₕₕ where L = Σₜ Lₜ and hₜ depends on Wₕₕ at every step

 Step 1 — Loss gradient w.r.t. output logits (standard cross-entropy)



 eᵧₜ is the one-hot for true class yₜ. Flows back to hₜ via Wₕᵧ.


 Step 2 — Gradient w.r.t. hₜ (receives gradient from Lₜ and hₜ₊₁)



 Two sources: output loss at t, and backprop from the NEXT state via Wₕₕ.


 Step 3 — Through tanh: multiply by local Jacobian diag(1 − hₜ²)



 If |hₜ| ≈ 1 (saturated), then 1−hₜ² ≈ 0 → gradient signal is killed.


 Step 4 — Accumulate gradient for Wₕₕ over all timesteps



 Wₕₕ is shared across time so gradients from ALL steps are SUMMED.
                                                                                                                  L2 · RNNs & BPTT
 Vanishing & Exploding Gradients

Key quantity: gradient of hₜ w.r.t. hₜ₋ₖ (k steps back)




  Vanishing Gradient                                          Gradient Clipping

                                                               •     if ||g|| > threshold:
                                                               •        g ← g × threshold / ||g||
                                                               •     Typical threshold: 1.0 or 5.0
   •     If the eigenvalues of Wₕₕ < 1 (or tanh saturates):
 As k → ∞, (λ·γ)^k → 0 if λ·γ < 1
 Gradient vanishes → no long-range learning                   Why Vanishing Is the Real Problem
                                                               •     Exploding: easy to detect (NaN/Inf), easy to fix (clipping)
                                                               •     Vanishing: silent failure — loss decreases but long-range
  Exploding Gradient                                                 patterns aren't learned
   •     If eigenvalues of Wₕₕ > 1: gradient norm grows        •     Root cause: repeated multiplication by Wₕₕ across many
         exponentially                                               steps
   •     Fix: gradient clipping — preserve direction, cap      •     Fix: LSTM/GRU gates control information flow → Lecture
         magnitude                                                   3
LEC TUR E 3     ·   MODULE        5




Gated Units: LSTMs & GRUs


•
•
•
    LSTM: cell state, forget gate, input gate, output gate
    All 6 LSTM equations derived
    Why the cell state creates a gradient highway
                                                             3
•   GRU: reset gate, update gate — 4 equations
•   LSTM vs GRU comparison
                                                                                                                         L3 · LSTMs & GRUs
LSTM: Motivation & the Gradient Highway

                                                                RNN vs LSTM gradient path:
 What we want
   •      Remember information across many timesteps without    Plain RNN:
          gradient decay
   •      Forget irrelevant past information adaptively
                                                                        tanh               tanh                tanh           tanh
   •      Selectively write new information to memory
   •      All controlled by the network itself
                                                                Gradient × Wₕₕ at every step → exponential decay
 Cell State — Constant Error Carousel
                                                                LSTM cell state:
   •      Hochreiter & Schmidhuber (1997): introduce a CELL
          STATE cₜ
                                                                         c₁                 c₂                  c₃             c₄
   •      cₜ flows through time with only ⊙ and + — no matrix
          multiply!
   •      Gates modulate what enters and exits the cell state   Gradient flows through additions — no decay!


Key gradient identity:                                            Why Addition Fixes Vanishing

                                                                    •         ∂cₜ/∂cₜ₋₁ = fₜ (element-wise, not a matrix product)
                                                                    •         If fₜ ≈ 1 over many steps: gradient magnitude ≈ 1^k = 1
                                                                    •         No repeated matrix multiplication → no exponential
If fₜ ≈ 1: gradient × 1 at every step — no decay!                             decay
                                                                                                                                                       L3 · LSTMs & GRUs
 LSTM: All 6 Equations

Inputs: xₜ ∈ ℝᵈ, hₜ₋₁ ∈ ℝᴴ, cₜ₋₁ ∈ ℝᴴ

  Forget gate fₜ — decides what to erase from cell state                             Input gate iₜ — decides which new values to store




  fₜ≈0: forget c ₜ₋₁. fₜ≈1: keep everything. [hₜ₋₁,x ₜ] = concatenation.             Controls how much of candidate ĉ ₜ gets written. iₜ≈0: ignore. iₜ≈1: fully write.



  Candidate cell ĉₜ — proposed new content (tanh like vanilla RNN)                   Cell state update — forget old + write new (the gradient highway)




  Proposes information from current input and previous hidden state. Gated by i ₜ.   No weight matrix — only additions and ⊙. Derivative w.r.t. c ₜ₋₁ is just fₜ.



  Output gate oₜ — decides how much cell state to expose                             Hidden state hₜ — filtered view of cell state




  Even if cell state carries important info, network can choose not to expose it.    Passed to next layer and next timestep. tanh squashes c ₜ values to (−1,1).
                                                                                                                                                             L3 · LSTMs & GRUs
 GRU: Gated Recurrent Unit

 GRU Motivation (Cho et al., 2014)
   •      LSTM has 4 parameter groups — expensive to train
   •      GRU merges cell state and hidden state into one vector
   •      Only 2 gates → 3 parameter groups instead of 4
   •      Often matches LSTM performance with less compute

GRU Equations — all 4 steps:

 Reset gate rₜ — how much to forget previous hidden state                                  Update gate zₜ — interpolation between past and candidate




 rₜ≈0: ignore past hₜ₋₁ when computing candidate (fresh start). rₜ≈1: full access.         A single gate does the work of both f and i in LSTM.



 Candidate hidden state h̃ₜ                                                                Hidden state — linear interpolation controlled by update gate




 Reset gate rₜ filters hₜ₋₁ before computing candidate. If rₜ=0, hₜ̃ depends only on xₜ.   zₜ=0: copy old state. z ₜ=1: fully replace with candidate. No separate cell state.
LEC TUR E 4    ·   MODULE 5




## Seq2Seq & Attention Mechanisms


•
•
•
    Encoder-decoder architecture
    The information bottleneck and why it fails
    Bahdanau (additive) attention — full derivation
                                                      4
•   Luong (multiplicative) attention
•   Scaled dot-product attention
                                                                                                        L4 · Seq2Seq & Attention
Seq2Seq: Encoder-Decoder Architecture

                                                                  Encoder-Decoder diagram:
 The Seq2Seq Task
  •        Map variable-length source to variable-length target
  •        E.g. machine translation, summarisation

Encoder:
                                                                                   ENCODER         DECODER         ŷ₁



                                                                        hᵉ₁          hᵉ₂     hᵉ₃    c              sₜ₁

 Final hidden state becomes context vector c = h_Tx
  Encodes the ENTIRE source into one fixed-size vector — the
                                                                         x₁           x₂     x₃
  bottleneck!

Decoder:
                                                                                                       L4 · Seq2Seq & Attention
 Bahdanau Attention: Full Derivation

Given: encoder states h₁ᵉ,…,hTₓᵉ and decoder state at step t: sₜ₋₁

 Step 1 — Alignment score eₜⱼ: how relevant is encoder position j for decoder step t?



 Wₐ ∈ ℝ^(A×H), Uₐ ∈ ℝ^(A×H), vₐ ∈ ℝᴬ are learned. Both sₜ₋₁ and hⱼᵉ mapped to common space via tanh.


 Step 2 — Softmax over scores → attention weights (soft distribution over source)



 αₜⱼ ∈ (0,1) and Σⱼ αₜⱼ = 1. Probability that decoder step t attends to encoder position j.


 Step 3 — Context vector cₜ: weighted sum of encoder hidden states



 Different cₜ at every decoder step! Unlike basic Seq2Seq which uses one fixed c. Dynamic context.


 Step 4 — Decoder update using dynamic context cₜ



 cₜ concatenated with previous token y ₜ₋₁ as decoder input. Direct access to relevant source parts.
                                                                                                                                    L4 · Seq2Seq & Attention
 Scaled Dot-Product Attention & Score Function Comparison

Vaswani et al. (2017) — Attention Is All You Need:




                                                                                 Name                    Score function          Params
  Why divide by √dₖ?

                                                                                 Additive (Bahdanau)     vₐᵀ tanh(Wₐs + Uₐh)     Wₐ, Uₐ, vₐ


                                                                                 Dot-product (Luong)     sᵀh                     none
Large dₖ → large dot products → softmax saturation → tiny gradients →
divide by √dₖ keeps variance ≈ 1
                                                                                 General (Luong)         sᵀ Wₐ h                 Wₐ only

  Attention Gradient — No Bottleneck
                                                                                 Scaled dot              qᵀk / √dₖ               Wq, Wk, Wv
                                                                                 (Transformer)


                                                                                 One question, many answers
Every encoder state hⱼᵉ gets gradient from ALL decoder steps (weighted by αₜⱼ)     All score functions ask: how similar is this query to this key?
                                                                                                                      Summary
Module 5 Summary

L1 — Embeddings                                                 L3 — LSTMs & GRUs

 •    One-hot → dense; Skip-gram SGNS objective                  •    LSTM: 6 equations; ∂cₜ/∂cₜ₋₁ = fₜ (gradient highway)
 •    P(o|c) = softmax(ṽₒᵀvᶜ); vector arithmetic = semantics     •    GRU: 4 equations; 25% fewer params; both fix vanishing




L2 — RNNs & BPTT
                                                                L4 — Seq2Seq & Attention

 •    hₜ = tanh(Wₕₕhₜ₋₁+Wₓₕxₜ+bₕ); shared weights across time
 •    ∂L/∂Wₕₕ = Σₜ δzₜhₜ₋₁ᵀ; (λ·γ)^k → 0 for vanishing           •    Encoder-decoder with fixed context c = hTₓᵉ
                                                                 •    Bahdanau: eₜⱼ=vₐᵀtanh(Wₐsₜ₋₁+Uₐhⱼ); cₜ=Σαₜⱼhⱼ
BPTT: 

## Deriving the Recurrence and the Weight Gradients
                 From ∂L/∂ht to ∂L/∂Whh




                                                         1 / 11
Setup: what we want to compute
For a vanilla RNN,
                                                                       T
                                                                       X
                 at = Wxh xt + Whh ht−1 + b,       ht = ϕ(at ),   L=         Lk .
                                                                       k=1
Ultimate goal:
                                              ∂L
                                                   .
                                             ∂Whh
Why do we first derive a recurrence for the state gradient?
                                      ∂L               ∂L
                                            or δt :=
                                     ∂ht               ∂at
Because Whh affects the loss through the hidden states over time. Once δt is known, the
weight gradient becomes simple.

                                               T
                                      ∂L    X
                                                  ⊤
                                          =   δt ht−1
                                     ∂Whh
                                               t=1                                        2 / 11
Step 1: which losses depend on ht ?

The hidden state ht affects:
     Lt directly,
     Lt+1 through ht+1 ,
     Lt+2 through ht+1 → ht+2 ,
     ...,
     LT through the whole future chain.
So
                               ∂L     ∂
                                   =     (Lt + Lt+1 + · · · + LT ).
                               ∂ht   ∂ht
Therefore
                               ∂L    ∂Lt    ∂
                                   =     +     (Lt+1 + · · · + LT ).
                               ∂ht   ∂ht   ∂ht
Key idea: the derivative is a sum because ht influences the loss through multiple branches.
                                                                                              3 / 11
Step 2: derive the recurrence for ∂L/∂ht

All future losses depend on ht only through ht+1 . So by the chain rule,

                             ∂                          ∂L ∂ht+1
                                (Lt+1 + · · · + LT ) =           .
                            ∂ht                        ∂ht+1 ∂ht

Hence
                                  ∂L    ∂Lt    ∂L ∂ht+1
                                      =     +
                                  ∂ht   ∂ht   ∂ht+1 ∂ht

Interpretation:
     first term = current-time contribution
     second term = future contribution propagated backward through the recurrence


                                                                                    4 / 11
Step 3: expand the local transition Jacobian
From
                    at+1 = Wxh xt+1 + Whh ht + b,    ht+1 = ϕ(at+1 ),
                                  ∂ht+1    ∂ht+1 ∂at+1
                                        =              .
                                   ∂ht     ∂at+1 ∂ht
Now
                             ∂at+1                  ∂ht+1
                                   = Whh ,                = Dt+1 ,
                              ∂ht                   ∂at+1
where
                          Dt+1 = diag ϕ′ (at+1,1 ), . . . , ϕ′ (at+1,n ) .
                                                                        

So
                                      ∂ht+1
                                            = Dt+1 Whh .
                                       ∂ht
Thus the state-gradient recurrence becomes
                                ∂L     ∂Lt    ∂L
                                    =      +       Dt+1 Whh .
                                ∂ht    ∂ht   ∂ht+1
                                                                             5 / 11
Step 4: define δt and get the standard BPTT recursion
Define the pre-activation error
                                                     ∂L
                                             δt :=       .
                                                     ∂at
Since ht = ϕ(at ),
                                                ∂L
                                         δt =       ⊙ ϕ′ (at ).
                                                ∂ht
Also, because at+1 = Whh ht + · · · ,
                                        ∂L            ⊤
                                                   = Whh δt+1 .
                                        ∂ht future
So
                                      ∂L    ∂Lt      ⊤
                                          =     + Whh  δt+1 .
                                      ∂ht   ∂ht
Multiplying elementwise by ϕ′ (at ) gives
                                                     
                                        ∂Lt
                               δt =         + Whh δt+1 ⊙ ϕ′ (at )
                                                ⊤
                                        ∂ht
                                                                    6 / 11
Step 5: now derive the gradient with respect to Whh
Since the same recurrent matrix is used at every time step,
                                              T
                                     ∂L    X ∂L ∂at
                                         =            .
                                    ∂Whh     ∂at ∂Whh
                                              t=1
Using δt = ∂L/∂at and
                                     at = Whh ht−1 + · · · ,
we get the local derivative
                                           ∂at
                                               ⇝ ht−1 .
                                          ∂Whh
                                                  T
                                       ∂L    X
                                                   ⊤
                                           =   δt ht−1
                                      ∂Whh
                                                  t=1

                                      T                        T
                               ∂L    X                  ∂L X
                                   =   δt xt⊤ ,            = δt .
                              ∂Wxh                      ∂b
                                      t=1                      t=1
                                                                     7 / 11
3-step unrolled RNN: direct expansion of ∂L/∂Whh

Let L = L1 + L2 + L3 . Since Whh is used at all three steps,
                           ∂L    ∂L ∂a1    ∂L ∂a2   ∂L ∂a3
                               =         +        +         .
                          ∂Whh   ∂a1 ∂Whh ∂a2 ∂Whh ∂a3 ∂Whh

Defining δt := ∂L/∂at gives

                              ∂L        ∂a1       ∂a2       ∂a3
                                  = δ1      + δ2      + δ3      .
                             ∂Whh      ∂Whh      ∂Whh      ∂Whh
Because at = Whh ht−1 + · · · ,

                                    ∂L
                                        = δ1 h0⊤ + δ2 h1⊤ + δ3 h2⊤
                                   ∂Whh

So the remaining task is: compute δ1 , δ2 , δ3 efficiently.
                                                                     8 / 11
3-step example: backward recursion for δ3 , δ2 , δ1
At the final step,
                                            ∂L3
                                          δ3 =  ⊙ ϕ′ (a3 ).
                                            ∂h3
At time 2, a2 affects both L2 and the future through a3 :
                                                   
                                       ∂L2
                               δ2 =        + Whh δ3 ⊙ ϕ′ (a2 ).
                                                 ⊤
                                       ∂h2
At time 1, the whole future is summarized by δ2 :
                                                  
                                       ∂L1
                                δ1 =       + Whh δ2 ⊙ ϕ′ (a1 ).
                                                ⊤
                                       ∂h1
So in general,
                                                          
                                          ∂Lt    ⊤
                               δt =           + Whh δt+1       ⊙ ϕ′ (at )
                                          ∂ht
This recurrence compresses all future paths into a single backward signal.
                                                                             9 / 11
Why the recurrence matters
Without the recurrence, the derivative at early time steps explodes into many explicit future
paths:

         a1 → h1 → a2 → h2 → L2 ,         a1 → h1 → a2 → h2 → a3 → h3 → L3 ,        ...

                             L1                      L2                      L3



                 a1          h1          a2          h2          a3          h3




                         Many future paths from early states

                                                            ⊤δ
The backward recursion summarizes these paths compactly in Whh t+1 .
                                                                                            10 / 11
Takeaway

 1   First derive the state-gradient recurrence:
                                     ∂L    ∂Lt    ∂L ∂ht+1
                                         =     +           .
                                     ∂ht   ∂ht   ∂ht+1 ∂ht
 2   Convert it to the standard pre-activation recursion:
                                                         
                                         ∂Lt
                                 δt =        + Whh δt+1 ⊙ ϕ′ (at ).
                                                   ⊤
                                         ∂ht
 3   Then the recurrent weight gradient is easy:
                                                   T
                                          ∂L    X
                                                      ⊤
                                              =   δt ht−1 .
                                         ∂Whh
                                                   t=1



                                                                      11 / 11
