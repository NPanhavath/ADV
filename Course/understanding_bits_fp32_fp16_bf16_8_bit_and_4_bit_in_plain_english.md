# The Ultimate Beginner's Guide to AI Numbers: Bits, FP32, FP16, BF16, 8-Bit, and 4-Bit

If words like `FP32`, `float16`, `bfloat16`, `8-bit`, and `4-bit` sound like alien computer code, **do not worry at all.** Almost everyone feels confused the first time they see them! 

This guide explains everything from zero, using everyday examples and super simple English.

---

## Part 1: Why Do We Even Care About Numbers in AI?

First, what actually *is* an AI model (like ChatGPT or the Qwen model you just trained)?

Inside the computer, an AI model is **not** a magic brain. It is simply a **giant spreadsheet of billions of numbers**. In AI, we call these numbers **weights** (or **parameters**).

* When you loaded **`Qwen2.5-3B`**, the **"3B"** stands for **3 Billion**.
* That means the model is literally a list of **3,000,000,000 decimal numbers** (like `0.0245`, `-1.8492`, `13.45678`).
* When you ask the AI a question in Khmer, your GPU multiplies your words by all 3 billion numbers to predict the best answer.

**The Big Problem:** 
Storing 3 billion numbers takes up a lot of space on your graphics card (GPU memory, called **VRAM**). If the numbers are too big and detailed, your GPU runs out of space and crashes!

To fix this, computer scientists asked: *"How can we write down these 3 billion numbers using less space?"*

---

## Part 2: What is a "Bit"? (The Light Switch Analogy)

Computers do not have fingers to count with, and they do not know the numbers `0` through `9`. 

Instead, a computer is made of billions of microscopic **light switches**.
* A switch can only be **OFF (`0`)** or **ON (`1`)**.
* **One single switch is called 1 bit.**

### How many numbers can you make with switches (bits)?
* **1 bit (1 switch):** Can only be `0` or `1` $\rightarrow$ **2 possible choices** ($2^1 = 2$).
* **2 bits (2 switches):** Can be `00`, `01`, `10`, or `11` $\rightarrow$ **4 possible choices** ($2^2 = 4$).
* **4 bits (4 switches):** Gives you **16 possible choices** ($2^4 = 16$).
* **8 bits (8 switches):** Gives you **256 possible choices** ($2^8 = 256$). *(Note: 8 bits is also called **1 Byte**!)*
* **16 bits (16 switches):** Gives you **65,536 possible choices** ($2^{16} = 65,536$).
* **32 bits (32 switches):** Gives you **4,294,967,296 (over 4 billion!) possible choices** ($2^{32}$).

### Think of Bits Like "Boxes" for Writing Numbers
Imagine I give you a row of empty boxes on a piece of paper to write down a price:
* If I give you **32 boxes (`32-bit`)**, you can write a super exact number like: `$13.45678912`
* If I only give you **4 boxes (`4-bit`)**, you don't have enough room for all those extra digits! You have to round it to just: `$13`

**Rule of Thumb:**
* **More bits** = More exact numbers, **but** uses **more GPU memory** and runs **slower**.
* **Fewer bits** = Rounded numbers, **but** uses **way less GPU memory** and runs **much faster**.

---

## Part 3: What Does "FP" or "Float" Mean? (`FP32`, `FP16`, `BF16`)

In your code and slides, you saw two words:
1. **`INT` (Integer):** Whole numbers with **no decimal point** (like `1`, `5`, `-12`, `71`).
2. **`FP` or `float` (Floating-Point):** Numbers **with a decimal point** (like `3.14`, `9.34`, `13.45678`). 
   * Why is it called "floating" point? Because the dot (`.`) can "float" or move around (e.g., `123.4` vs `1.234`).

Inside a floating-point number, the computer splits its switches (bits) into **three jobs**:

$$\underbrace{\text{1. Sign}}_{\text{Plus or Minus (+ / -)}} \quad \Big| \quad \underbrace{\text{2. Size / Range (Exponent)}}_{\text{How HUGE the number can be}} \quad \Big| \quad \underbrace{\text{3. Detail / Precision (Fraction)}}_{\text{How many digits after the decimal dot}}$$

Let's look at the three types of decimal numbers used in AI:

---

### 1. `FP32` (`float32` — 32-Bit Floating Point)
* **What the name means:** **F**loating **P**oint using **32** bits (switches).
* **Analogy:** A **4K Ultra-HD photo**. Every tiny detail is crystal clear, but the file size is huge.
* **How it works:** It uses 32 switches per number (1 for $+/-$, 8 for Size, 23 for Decimal Detail). It remembers about **7 decimal places** (like `13.4567812`).
* **How much GPU memory it costs:** Every 1 billion model parameters takes **4 GB of VRAM**.
  * For your **3 Billion (`Qwen-3B`)** model: $3 \times 4\text{ GB} = \mathbf{12\text{ GB of VRAM}}$ just to open the model! That almost fills your entire 14.5 GB Google Colab GPU before training even starts.

---

### 2. `FP16` (`float16` — 16-Bit Floating Point)
* **What the name means:** **F**loating **P**oint cut in half to **16** bits (switches).
* **Analogy:** A **1080p HD photo**. It takes **half the storage space** of 4K, and to the human eye, it looks almost identical!
* **How much GPU memory it costs:** Every 1 billion parameters takes **2 GB of VRAM** (half of `FP32`).
  * For your **3 Billion (`Qwen-3B`)** model: $3 \times 2\text{ GB} = \mathbf{6\text{ GB of VRAM}}$.
* **The One Catch with `FP16` (The Size Limit Problem):**
  * To shrink from 32 bits to 16 bits, `FP16` only kept **5 switches for the Size** and **10 switches for Decimal Detail**.
  * Because it only has 5 switches for Size, the **largest number `FP16` can ever count up to is `65,504`**.
  * If the AI is doing math during training and a number accidentally hits `70,000`, `FP16` panics! It cannot hold `70,000`, so it turns the number into `Infinity` (`NaN` = Not a Number), and your training crashes.

---

### 3. `BF16` (`bfloat16` — 16-Bit "Brain" Floating Point)
* **What the name means:** **B**rain **F**loating **P**oint using **16** bits.
* **Who invented it and why?** Google's AI research team ("Google Brain") got tired of `FP16` crashing whenever a number went above `65,504`.
* **How Google fixed it:**
  * Google realized something cool about AI brains: *AI models don't care if `3.1415` is rounded to `3.14`, as long as the model can still hold gigantic numbers without crashing!*
  * So, `bfloat16` uses 16 bits, **just like `FP16`**, so it takes the exact same small memory (**6 GB for a 3B model**).
  * **BUT** inside those 16 bits, it gives **8 switches to Size** (the exact same as `FP32`!) and only **7 switches to Decimal Detail**.
  * Now, `bfloat16` can hold gigantic numbers (up to $340,000,000,000,000,000,000,000,000,000,000,000,000$) and **never crashes**, while only using 16 bits of memory!
* **Why did your notebook say `Bfloat16 = FALSE`?**
  * `bfloat16` was invented after older graphics cards were already made.
  * Google Colab's free GPU (**Tesla T4**) was made in 2018, before `bfloat16` existed on standard GPUs.
  * That is why we wrote these two smart lines in **Cell 5** of your notebook:
    ```python
    fp16 = not is_bfloat16_supported(), # Use FP16 if the GPU is older (like Tesla T4)
    bf16 = is_bfloat16_supported(),     # Use BF16 if the GPU is newer (like RTX 3050 / A100)
    ```
  * Your code automatically checked your GPU, saw it was a Tesla T4, and safely used `fp16`!

---

## Part 4: What is Quantization? (`8-Bit` and `4-Bit` / `NF4`)

Look at **Slide 12** of your lesson PDF. You see a picture where:
* `13.45678` becomes `13`
* `9.34` becomes `9`
* `2.45678` becomes `2`

**That process is called Quantization!**

**Quantization** is just a fancy university word for **"rounding numbers so they fit into smaller boxes (fewer bits)."**

### The Crayon Box Analogy
Imagine an artist painting a portrait:
* **`FP32` / `FP16`** is a giant box of **65,536 crayons**. You have 500 slightly different shades of blue.
* **`8-Bit` (`INT8`)** is a box of **256 crayons** ($2^8 = 256$). Whenever the original painting uses "Sky Blue #42", you swap it with the closest crayon in your 256-pack. The picture still looks great, and the box is much smaller!
* **`4-Bit` (`NF4`)** is a tiny pocket box of **only 16 crayons** ($2^4 = 16$). Every single number in the 3-billion-parameter AI model gets mapped to one of the **16 closest values**.

### Wait... What is `NF4` (`bnb_4bit_quant_type="nf4"` on Slide 13)?
In **Slide 13**, you saw `quant_type="nf4"`. 
* **`NF4` stands for NormalFloat 4-bit.**
* If you only get **16 crayons (16 buckets)** for your 3 billion AI numbers, how do you choose which 16 numbers to put in your box?
* Scientists discovered that 95% of all numbers inside an AI model are clustered tightly around `0` (like `-0.05`, `0.01`, `0.12`), and very few numbers are far away (like `-2.0` or `+2.0`). This bell-curve shape is called a **Normal Distribution**.
* So **`NF4` spaces out the 16 buckets unevenly**: it puts most of the 16 buckets super close to `0` (where most of the AI's numbers live) and spreads the outer buckets further apart. That way, even with only 4 bits, the AI loses almost zero intelligence!

### How Much GPU Memory Does 4-Bit Save?
Look at how your **Qwen 2.5 (3 Billion Parameters)** model shrinks as we reduce the bits:

| Number Type | Bits per Parameter | Memory for 1 Number | Total VRAM for Qwen-3B (3 Billion Numbers) | Fits on Free Colab GPU? |
| :--- | :--- | :--- | :--- | :--- |
| **`FP32`** | 32 bits | 4 Bytes | **~12.0 GB** | Barely (crashes when training starts) |
| **`FP16` / `BF16`** | 16 bits | 2 Bytes | **~6.0 GB** | Fits, but tight during training |
| **`8-Bit` (`INT8`)** | 8 bits | 1 Byte | **~3.0 GB** | Yes, easily! |
| **`4-Bit` (`NF4`)** | 4 bits | 0.5 Bytes | **~1.5 GB to 2.0 GB** | **Tiny! Uses almost no memory!** |

Because you used `load_in_4bit = True` in **Cell 2**, your entire 3-billion-parameter model shrank from **12 GB down to ~2 GB**!

---

## Part 5: How Do 4-Bit and 16-Bit Work Together in QLoRA? (Slide 13 & 14)

Wait—if we loaded the model in **4-bit** in Cell 2 (`load_in_4bit = True`), why did we *also* set `fp16 = True` (16-bit) in Cell 5? And why does **Slide 13** have both `load_in_4bit` AND `compute_dtype=torch.float16`?

Here is the secret of how **QLoRA** works, using a **ZIP File Analogy**:

1. **Storage is 4-Bit (The Zipped File):**
   * When the 3 billion frozen base model weights are sitting in your GPU memory waiting to be used, they are stored in **4-bit (`NF4`)** so they only take up ~2 GB of space.
   * However, **4-bit numbers are too rounded to do accurate training math** (if you try to add a tiny training update like `+0.0002` to a 4-bit number, it just gets rounded to `0` and the model learns nothing!).
2. **LoRA Adapters & Math are 16-Bit (Unzipping to Do Work):**
   * The small **LoRA adapters** you added in Cell 4 (`0.96%` of the model) are stored in crisp **16-bit (`FP16` or `BF16`)**.
   * When a Khmer sentence flows through a layer of the model during training, the GPU temporarily **"unzips" (dequantizes)** that specific 4-bit layer into **16-bit (`float16`)**, does the accurate multiplication and learning in 16-bit, updates the 16-bit LoRA adapter, and immediately throws away the temporary unzipped copy!

**In short:**
* **`4-bit` = Storage format** (keeps GPU memory usage super low).
* **`float16` / `bfloat16` = Working/Math format** (`compute_dtype`—keeps the math and LoRA learning accurate).

---

## Quick Cheat Sheet Summary

Whenever you see these words in code or slides, just translate them in your head like this:

* **`Bit`:** A single `0` or `1` memory slot. More bits = more exact numbers, but uses more GPU memory.
* **`FP32` (`float32`):** Full-size decimal numbers (32 bits). Super accurate, but way too heavy for most GPUs.
* **`FP16` (`float16`):** Half-size decimal numbers (16 bits). Fast and uses half the memory, used on older GPUs like Google Colab's free **Tesla T4**.
* **`BF16` (`bfloat16`):** Modern 16-bit decimal numbers invented by Google. Same size as `FP16`, but can hold gigantic numbers without crashing. Used on newer GPUs (like RTX 30-series/40-series and A100).
* **`Quantization`:** Rounding high-detail numbers (`FP32`/`FP16`) into low-bit numbers (`8-bit` or `4-bit`) so huge AI models fit on small GPUs.
* **`8-Bit` (`load_in_8bit=True`):** Shrinks model memory by **75%** compared to `FP32` (256 possible values per weight).
* **`4-Bit` / `NF4` (`load_in_4bit=True`):** Shrinks model memory by **87.5%** compared to `FP32` (16 smart buckets per weight). This is the **"Q"** in **QLoRA**!