# Step-by-Step LoRA Math Walkthrough (Toy Example)

This walkthrough simulates the math running behind the notebook's training loop[cite: 2]. We shrink the model down to single scalar values and compute **2 complete training iterations** by hand.

---

## 1. Setup & Initial Parameters

In LoRA, the base model weights ($W_0$) are frozen[cite: 2]. The forward pass is:

$$\hat{y} = \underbrace{W_0 \cdot x}_{\text{Frozen Base}} + \underbrace{B \cdot A \cdot x}_{\text{Trainable LoRA}}$$

* **Input token ($x$):** $2$
* **Target output ($y$):** $10$
* **Frozen base weight ($W_0$):** $3$
* **LoRA Matrix $B$ ($B_{\text{init}}$):** $0$ *(LoRA initializes $B$ to zero so the initial model output is unchanged)*
* **LoRA Matrix $A$ ($A_{\text{init}}$):** $0.5$ *(Initialized randomly)*
* **Learning Rate ($\eta$):** $0.1$
* **Loss Function:** $L = \frac{1}{2}(\hat{y} - y)^2$

---

## 2. Iteration 1

### Step 1: Forward Pass (Predict Output)
* **Base prediction:** $W_0 \cdot x = 3 \times 2 = 6$
* **LoRA adaptation:** $B \cdot A \cdot x = 0 \times 0.5 \times 2 = 0$
* **Combined Output ($\hat{y}_1$):**
  $$\hat{y}_1 = 6 + 0 = 6$$

---

### Step 2: Compute Loss
* **Error:** $\hat{y}_1 - y = 6 - 10 = -4$
* **Loss ($L_1$):**
  $$L_1 = \frac{1}{2}(-4)^2 = 8.00$$

---

### Step 3: Backward Pass (Compute Gradients)
Base weight $W_0$ is frozen, so $\frac{\partial L}{\partial W_0} = 0$[cite: 2]. Gradients are calculated only for $B$ and $A$[cite: 2]:

* **Gradient for $B$:**
  $$\frac{\partial L}{\partial B} = (\hat{y}_1 - y) \cdot (A \cdot x) = (-4) \times (0.5 \times 2) = -4.0$$

* **Gradient for $A$:**
  $$\frac{\partial L}{\partial A} = (\hat{y}_1 - y) \cdot B \cdot x = (-4) \times 0 \times 2 = 0.0$$

---

### Step 4: Weight Update (Gradient Descent)
$$\text{New Weight} = \text{Old Weight} - (\eta \times \text{Gradient})$$

* **Update $B$:**
  $$B_1 = 0 - (0.1 \times -4.0) = 0 + 0.4 = \mathbf{0.4}$$

* **Update $A$:**
  $$A_1 = 0.5 - (0.1 \times 0.0) = \mathbf{0.5}$$

---

## 3. Iteration 2

Using updated parameters: $B = 0.4$, $A = 0.5$.

### Step 1: Forward Pass (Predict Output)
* **Base prediction:** $W_0 \cdot x = 3 \times 2 = 6$
* **LoRA adaptation:** $B \cdot A \cdot x = 0.4 \times 0.5 \times 2 = 0.4$
* **Combined Output ($\hat{y}_2$):**
  $$\hat{y}_2 = 6 + 0.4 = \mathbf{6.4}$$
  *(The output moved closer to the target of 10).*

---

### Step 2: Compute Loss
* **Error:** $\hat{y}_2 - y = 6.4 - 10 = -3.6$
* **Loss ($L_2$):**
  $$L_2 = \frac{1}{2}(-3.6)^2 = \mathbf{6.48}$$
  *(Loss decreased from 8.00 to 6.48, mirroring the decreasing validation loss in the notebook table[cite: 2]).*

---

### Step 3: Backward Pass (Compute Gradients)
* **Gradient for $B$:**
  $$\frac{\partial L}{\partial B} = (-3.6) \times (0.5 \times 2) = -3.6$$

* **Gradient for $A$:**
  $$\frac{\partial L}{\partial A} = (-3.6) \times 0.4 \times 2 = -2.88$$

---

### Step 4: Weight Update
* **Update $B$:**
  $$B_2 = 0.4 - (0.1 \times -3.6) = 0.4 + 0.36 = \mathbf{0.76}$$

* **Update $A$:**
  $$A_2 = 0.5 - (0.1 \times -2.88) = 0.5 + 0.288 = \mathbf{0.788}$$

---

## 4. Key Takeaways

1. **Base Model Remains Untouched:** $W_0$ stayed at $3$ throughout both steps[cite: 2].
2. **LoRA Adapts the Output:** As $B$ and $A$ adjust, the additive delta ($B \cdot A \cdot x$) grows to steer the frozen prediction toward the training target[cite: 2].
3. **Loss Decreases:** The error dropped from $-4.0$ to $-3.6$, reducing total loss from $8.00$ to $6.48$[cite: 2].

--- 

## Detailed Explanations & FAQ

---

### 1. What Does "Frozen" Mean?

In machine learning, **frozen** means **locked and unchangeable**[cite: 2].

* **Standard Training (Full Fine-Tuning):** The computer calculates gradients for all 3.1 billion parameters in Qwen2.5 and updates all of them[cite: 2]. This requires massive amounts of GPU memory (VRAM).
* **LoRA / QLoRA (Frozen Base):** The base model weights ($W_0$) are marked as `requires_grad = False`[cite: 2]. The computer loads them into GPU VRAM in 4-bit precision and promises never to alter them[cite: 2]. 
* **Analogy:** Think of the base model as a **printed textbook** in a library. You are not allowed to erase or rewrite any printed words (they are frozen). Instead, you stick **yellow sticky notes** (LoRA matrices $A$ and $B$) onto the pages[cite: 2]. When answering questions, you read the textbook plus your sticky notes[cite: 2]. Only the notes get edited or updated[cite: 2].

---

### 2. Is $\hat{y}$ (Prediction) Predicting Actual Words?

**Yes, but through numbers first.** Computers cannot do math on letters, so words are represented as numerical scores called **logits**.

* In our paper math example, $\hat{y} = 6$ was just a simple scalar output to keep the math easy.
* In a real LLM like Qwen2.5, the final output layer produces a list of ~152,000 scores (one score for every possible token in its vocabulary).
* $\hat{y}$ represents these predicted scores:
  * For example, the model scores candidates for the next token:
    * `"សួស្តី"` (Hello): score $1.2$
    * `"រាជធានី"` (Capital): score $8.9$  $\leftarrow$ **Highest score ($\hat{y}$)**
    * `"ចេក"` (Banana): score $-3.4$
* The model picks the token with the highest score, and the tokenizer converts that token ID back into the human-readable word `"រាជធានី"`.

---

### 3. What Does Input Token $(x) = 2$ Mean? (Real Example)

Before an AI reads any sentence, the **tokenizer** cuts the sentence into small pieces called tokens and turns each piece into an ID number.

#### Real Example:
Imagine our vocabulary has just 4 words:
* Token `0` = `"The"`
* Token `1` = `"capital"`
* Token `2` = `"is"`
* Token `3` = `"Phnom Penh"`

If the model is reading the word `"is"`, the tokenizer feeds the number **$2$** into the network. 

* In our toy walkthrough, $x = 2$ represents that input token ID.
* *(Note: Inside a real neural network, the integer $2$ is converted into a vector of numbers called an embedding, but on paper, we simplified that entire vector down to the single number $2$).*

---

### 4. Based on Our Notebook Project, How Many Iterations Did We Do?

We did exactly **57 iterations (training steps)**[cite: 2].

Here is how the training loop calculated that number[cite: 2]:

* **Dataset Size:** $450$ training examples (after splitting $90\%$ train and $10\%$ validation from 500 total rows)[cite: 2].
* **Per-Device Batch Size:** $2$ examples processed per micro-step[cite: 2].
* **Gradient Accumulation Steps:** $4$ micro-steps gathered before running one optimizer update[cite: 2].
- **Effective Batch Size:**[cite: 2]

  $$\text{Effective Batch Size} = 2 \times 4 = 8 \text{ examples per iteration}$$

- **Number of Epochs:** 1 full pass over the dataset.[cite: 2]

- **Total Iterations (Steps):**[cite: 2]

  $$\text{Total Steps} = \frac{450 \text{ examples}}{8 \text{ examples per step}} = 56.25 \approx 57 \text{ steps}$$

During those **57 iterations**, the model performed 57 forward passes, computed 57 backward gradient updates on the LoRA parameters, and finished training in **2 minutes and 40 seconds** on the Colab T4 GPU[cite: 2].

Pre-training and fine-tuning are two completely different stages in the life of an AI model.