# Qwen2.5-3B Khmer QLoRA Fine-Tuning with Unsloth

This notebook performs parameter-efficient fine-tuning (PEFT / QLoRA) on `Qwen2.5-3B-Instruct` using a Khmer instruction dataset on a single Google Colab GPU (Tesla T4).

---

### Step 1: Install Unsloth
- Installs the `unsloth` library and pulls the latest GitHub nightly build to ensure compatibility with Torch and CUDA.

### Step 2: Load Model & Tokenizer in 4-bit Precision
- **Model:** `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (pre-quantized 4-bit base model).
- **Function:** `FastLanguageModel.from_pretrained(...)`.
- **Purpose:** Loads both the model and its matching tokenizer into GPU VRAM using ~3–4 GB of memory instead of the 7 GB+ needed for FP16.

### Step 3: Dataset Preparation & Chat Templating
- **Dataset:** `Pisethan/khmer-instruct-dataset` (500 prompt-response pairs).
- **Mapping:** Reformats raw dictionary keys (`prompt`/`response` or `instruction`/`output`) into a standardized conversation list:
  - `system`: Sets the AI role ("អ្នកគឺជាជំនួយការ AI ដ៏ឆ្លាតវៃ។").
  - `user`: Input question.
  - `assistant`: Desired output.
- **Chat Template:** Converts the structured messages into Qwen's ChatML format (`<|im_start|>role ... <|im_end|>`) using `tokenizer.apply_chat_template()`.
- **Train/Test Split:** Splits 90% (450 examples) for training and 10% (50 examples) for validation.

### Step 4: Attach LoRA Adapters (PEFT)
- **Function:** `FastLanguageModel.get_peft_model(...)`.
- **Rank ($r=16$) & Alpha ($16$):** Freezes the 3-billion-parameter base model and attaches small trainable adapter matrices ($A$ and $B$) across all attention (`q, k, v, o`) and MLP layers (`gate, up, down`).
- **Efficiency:** Trains only **~29.9M parameters** (0.96% of the entire model), dramatically reducing memory and compute cost.

### Step 5: Configure Hyperparameters & Trainer
- Uses Hugging Face TRL's `SFTConfig` and `SFTTrainer`:
  - **Optimizer:** `adamw_8bit` (cuts optimizer VRAM usage).
  - **Effective Batch Size:** 8 (`batch_size=2` $\times$ `gradient_accumulation_steps=4`).
  - **Learning Rate:** $2 \times 10^{-4}$ with linear decay.
  - **Epochs:** 1 full pass (57 optimization steps).

### Step 6: Training Execution
- `FastLanguageModel.for_training(model)` activates gradient calculation on the LoRA weights.
- `trainer.train()` executes the 57 training steps in ~2 minutes and 40 seconds. Both training and validation losses decrease steadily (from ~0.039 down to ~0.015).

### Step 7: Saving & Inference Testing
- Saves the lightweight LoRA adapter checkpoint to disk (`Qwen2.5-Khmer-Unsloth/final_checkpoint`).
- Switches the model to inference mode via `FastLanguageModel.for_inference(model)`.
- Tests the model with a Khmer test query:
  - *"តើប្រទេសកម្ពុជាមានរាជធានីឈ្មោះអ្វី? សូមពន្យល់ឱ្យបានខ្លី។"* (*"What is the capital of Cambodia? Explain briefly."*)
- Generates text with `model.generate()` and decodes the token IDs back into readable Khmer text.