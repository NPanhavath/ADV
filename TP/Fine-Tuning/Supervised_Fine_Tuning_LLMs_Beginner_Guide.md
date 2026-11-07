# Supervised Fine-Tuning (SFT) for Large Language Models

> **A beginner-friendly guide to the full slide deck**  
> Based on *LLM Model Fine-Tuning* by Vann Visal (34 slides)

---

## How to use this guide

This guide follows the slide deck from the basic idea of fine-tuning through data preparation, training, checking results, and saving model checkpoints. Each section explains the ideas in plain English and adds context where that helps make the slides easier to understand.

### The full process at a glance

```text
Choose a pretrained model
        ↓
Load its tokenizer / processor
        ↓
Prepare and format example conversations
        ↓
Configure efficient training (often LoRA / PEFT)
        ↓
Train and monitor loss
        ↓
Save, reload, evaluate, or deploy a checkpoint
```

---

## 1. What is fine-tuning?

A **pretrained model** is a model that has already learned broad patterns from a very large amount of data. **Fine-tuning** means continuing its training with a smaller collection of examples focused on a particular task or style.

For example, a general language model could be fine-tuned to answer questions in a particular domain or to follow a particular conversation style. The model starts with what it already knows; training adjusts it to perform better on the new task.

Fine-tuning is not training from nothing. It reuses the pretrained model’s learned knowledge and adapts it.

---

## 2. Select and load the model

### What is a model loader?

A **model loader** is code that brings a pretrained model into computer memory. It loads the model architecture (the structure of the model) and its learned weights (the numerical values it uses). Once loaded, the model can be used to generate answers, or it can be fine-tuned.

### Model-specific loaders

A model-specific loader is for a known architecture. In the Transformers library, examples include:

```python
BertForSequenceClassification.from_pretrained("bert-base-uncased")
GPT2LMHeadModel.from_pretrained("gpt2")
T5ForConditionalGeneration.from_pretrained("t5-small")
```

Choose a specific loader when you know which model architecture you need.

### AutoModel loaders

An `AutoModel` loader chooses a compatible architecture from the model name or identifier. This can make it easier to switch models without changing the loader class each time.

```python
AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
AutoModelForTokenClassification.from_pretrained("roberta-base")
AutoModelForCausalLM.from_pretrained("gpt2")
```

The task-specific class matters: for example, causal language modeling and sequence classification are different jobs, so they use different model variants.

### Load a processor too

A model often needs a **processor** to turn its input into a form it can use. For text-only models, this is commonly a tokenizer. Other models may need an image processor, an audio processor, or a multimodal processor that handles more than one kind of input.

The slides describe `AutoProcessor` as the automatic way to choose the suitable processor for a model. Depending on the model, a processor may combine tokenization, feature extraction, image processing, or audio/text processing.

> **Key idea:** The model handles the learned task; the tokenizer or processor prepares the input in the format that model expects.

---

## 3. Quantization: using less memory

Model weights are numbers. A number can be stored at different levels of precision (how much detail is kept). **Quantization** stores weights or activations using fewer bits than a high-precision format. This can reduce memory use and sometimes improve speed, which is useful when working with large models or limited hardware.

The slides mention FP32, FP16, and INT8 as examples of numerical formats. They also show configuration options associated with the `BitsAndBytesConfig` family:

- `load_in_8bit=True` asks to load weights in 8-bit form.
- `bnb_8bit_use_double_quant=True` requests an additional quantization step intended to improve efficiency.
- `bnb_8bit_quant_type="nf4"` names a quantization type (NF4 is commonly used in 4-bit workflows).
- `bnb_8bit_compute_dtype=torch.float16` sets the numeric type used for calculations during the forward pass.

The slides also note that a model can be loaded with a quantization configuration or with a `load_in_4bit` option. These settings are examples, not a universal recipe: supported combinations depend on the library version, model, hardware, and training method. In particular, check that the option names and bit-width match the configuration you intend to use.

---

## 4. Prepare the training data

Training data must be structured so that the tokenizer, model, and training pipeline can all interpret it. Data preparation is not just about collecting examples: examples must be consistently organized and converted to the model’s expected input format.

### Start with raw data

The slides list JSON, CSV, and text as possible raw formats. The choice is flexible. What matters is that each example’s information is easy to retrieve. A key-value JSON structure is one convenient choice, and datasets in supported formats can be loaded with tools such as Hugging Face `load_dataset`.

A raw example might contain separate fields for a question and its answer. For a chat model, those fields need to be turned into a conversation structure next.

### Convert examples to conversation format

A conversation format represents a dialogue as an ordered set of messages. Each message usually has a role (who is speaking, such as `user` or `assistant`) and content (what they said). This makes the order and speaker of each turn clear, especially for multi-turn conversations.

A model’s expected conversation format is not always the same as another model’s. The slides recommend checking:

1. The model’s Hugging Face model page or description.
2. The model’s printed chat template.
3. The training dataset’s format.

### Apply the chat template

A **chat template** turns structured messages into one text sequence with the special markers and layout expected by a particular chat model. The Transformers function `apply_chat_template()` performs this conversion for compatible models.

This step matters because a model was trained to recognize a specific format. A conversation that looks clear to a person may not be correctly understood by the model if its expected markers or layout are missing.

### Use a data collator to build batches

A **data collator** prepares examples for the model in batches. A batch is a group of examples processed together. Depending on the collator and pipeline, it can:

- Tokenize text if it has not already been tokenized.
- Pad examples so they have compatible lengths in a batch.
- Add or prepare masks and labels used during training.
- Apply chat templates in collators designed to do that.

Collators may come from libraries such as Transformers or Unsloth, or they can be custom functions. Usually, you pass the collator to the trainer as a setting; it handles the batch preparation when training runs.

> **Data path:** raw records → structured messages → model’s chat template → tokenized and batched examples.

---

## 5. Make training efficient with PEFT and LoRA

### PEFT

**PEFT** means **Parameter-Efficient Fine-Tuning**. Instead of updating every parameter in a large model, PEFT methods update a smaller set. This can make fine-tuning faster and use less memory, which is especially helpful for large language models.

### LoRA

**LoRA** (Low-Rank Adaptation) is one PEFT method. It keeps the original model weights frozen (unchanged) and adds small trainable matrices to selected layers. Training updates these small additions rather than all the original weights.

The slides explain the idea with a matrix example:

- Original weight matrix `W`: `512 × 512` = **262,144** values.
- LoRA matrix `A`: `512 × 8` = **4,096** values.
- LoRA matrix `B`: `8 × 512` = **4,096** values.
- Together, `A` and `B` have **8,192** trainable values.

The update is written as:

```text
W′ = W + AB
```

The product `AB` has the same outer dimensions as `W` (`512 × 512`), but it is made from two much smaller matrices. With rank `r = 8`, the example trains 8,192 values instead of 262,144—about **97% fewer trainable parameters** for that matrix.

The rank is the small middle dimension (8 in this example). A smaller rank means fewer trainable values; the best rank depends on the task and model.

### Choose LoRA target modules

LoRA must be attached to particular layers or modules in the model. The slide suggests printing the model (`print(model)`) to inspect its module names and identify candidate targets. The correct target names depend on the architecture, so do not assume that names from one model will work for another.

### Configure LoRA and the trainer

The slide deck includes sections on LoRA configuration and trainer configuration, but does not provide readable parameter-by-parameter settings in the extracted slide text. In practice, LoRA settings describe where and how adapters are added; trainer settings describe how training proceeds (for example, its data, schedule, and batch handling). Choose values for the specific model, hardware, and training goal.

---

## 6. Start supervised training

After preparing the model and training configuration, the slides describe passing the required pieces to a trainer:

- The model.
- The processor or tokenizer.
- Training configuration.
- The data collator.
- The training dataset.
- An optional evaluation dataset.

Then call:

```python
trainer.train()
```

This starts the training loop. The trainer repeatedly prepares batches, asks the model to make predictions, compares predictions with the expected answers, and updates the trainable parameters.

**Supervised** training means that the examples include a desired answer or target. The model learns from these supplied examples rather than discovering the target on its own.

---

## 7. Diagnose training with loss

**Loss** is a score that measures how different the model’s predictions are from the expected answers. Lower loss generally means the predictions are closer to the training targets, but training loss alone does not tell the whole story.

### Training loss

Training loss is measured on the examples used to train the model. A decrease usually means the model is learning those examples. If it becomes very low, that does not prove the model will do well on new examples: it may have memorized the training data.

### Validation loss

Validation loss is measured on separate examples that are held back from training. It helps show whether the model can generalize—perform well on examples it did not train on.

| Training loss | Validation loss | Possible meaning |
|---|---|---|
| Decreasing | Also low / decreasing | The model may be learning useful patterns. |
| Low | Much higher | Possible **overfitting**: it does well on training data but poorly on unseen data. |
| High | High | Possible **underfitting**: it may not have learned enough yet, or the setup/data may need attention. |

These patterns are clues, not absolute proof. Consider them alongside example outputs and the training setup.

---

## 8. Track experiments with Weights & Biases

**Weights & Biases (W&B, also written WandB)** is a tool for recording and viewing machine-learning experiments. It can display metrics such as training loss over time, helping you compare runs and understand what happened.

The slides say that setting `report_to="wandb"` in the trainer configuration sends training metrics to W&B. They also advise initializing W&B before calling `trainer.train()`; the first use may ask you to log in. You can choose whether to use an experiment-tracking service, depending on your project.

---

## 9. Save and use model checkpoints

A **checkpoint** is a saved state of the model at a point during training. Depending on how it is saved, it can include model weights, optimizer state, the training step, loss, and other metadata.

Checkpoints are useful because they let you:

- Resume training after an interruption.
- Keep intermediate progress rather than starting over.
- Compare model versions from different training stages.
- Choose a version to use for inference (generating predictions or responses).

The slides show two ways to work with a trained checkpoint: load the checkpoint directly, or load the original base model and apply the checkpoint’s adapter to it. The adapter approach is relevant to methods such as LoRA; the adapter needs the base model it was trained with.

---

## 10. Important terms, in plain English

| Term | Meaning |
|---|---|
| **Pretrained model** | A model that has already learned from a large dataset. |
| **Fine-tuning** | Continuing training on focused examples to adapt a model. |
| **Weight / parameter** | A learned number inside the model. |
| **Tokenizer** | Converts text into units the model can process. |
| **Processor** | Prepares inputs, such as text, images, or audio, for a model. |
| **Quantization** | Stores model numbers at lower precision to reduce resource use. |
| **Conversation format** | A structured list of messages and speakers. |
| **Chat template** | The model-specific layout used to turn messages into input text. |
| **Data collator** | Prepares examples and groups them into batches for training. |
| **PEFT** | Fine-tuning methods that update only a small part of a model. |
| **LoRA** | A PEFT method that trains small low-rank matrices while keeping base weights fixed. |
| **Loss** | A measure of prediction error against expected answers. |
| **Validation** | Checking performance on examples not used for training. |
| **Overfitting** | Learning training examples too closely and doing worse on new examples. |
| **Underfitting** | Not learning enough to perform well, even on the task data. |
| **Checkpoint** | A saved model state from a training run. |

---

## 11. A practical mental checklist

Before training, ask:

- **Model:** Did I load a model for the task I want to do?
- **Processor:** Am I using the tokenizer or processor that belongs with it?
- **Quantization:** Do the precision settings match my hardware and chosen method?
- **Data:** Are examples clean, consistent, and structured correctly?
- **Chat format:** Does the conversation match this model’s expected template?
- **Batch preparation:** Is the collator preparing inputs, padding, and labels correctly?
- **Training method:** Am I updating all weights, or using a method such as LoRA?
- **Evaluation:** Do I have separate validation examples to check generalization?
- **Monitoring:** Am I checking both training and validation loss?
- **Saving:** Am I keeping checkpoints I may need to resume or compare?

---

## Main takeaway

Supervised fine-tuning adapts a pretrained model using examples that show the desired behavior. A successful workflow depends on more than calling the trainer: choose a suitable model and processor, format data the way the model expects, prepare batches correctly, select an appropriate training method, monitor validation performance, and save useful checkpoints. PEFT methods such as LoRA can make this process more practical by reducing how many parameters need to be trained.

---

*Source: “LLM Model Fine-Tuning,” presentation by Vann Visal, 34 slides. The slides include visual configuration/code pages whose exact settings are not represented in the extracted text; this guide explains their surrounding concepts without guessing those specific values.*

--- 

# Lesson 72: Tokenizers: How LLMs Convert Text to Numbers[cite: 13]

> **Key idea:** Models cannot process words or characters directly; tokenizers bridge human language and neural networks by fragmenting text into subwords (tokens) and mapping them to discrete vocabulary integers (token IDs).

---

## 1. The Two-Step Conversion Process

The transition from human-readable text to model input occurs in two distinct phases:

```
Natural Language Text ──> [ 1. Tokenization ] ──> Subword Tokens ──> [ 2. Numerical Mapping ] ──> Token IDs
("I love coding")         (Chunk into pieces)     ['I', ' love', ' coding'] (Vocabulary Index)    [40, 1842, 10254]
```

### Terminology Precision

| Term | What it is | Example |
| :--- | :--- | :--- |
| **Natural Language** | Unstructured characters, words, and sentences. | `"AI Engineering"` |
| **Token** | A text fragment, word, or subword chunk produced by the tokenizer. | `["AI", " Engineer", "ing"]` |
| **Token ID** | The unique integer index assigned to a specific token in the model's vocabulary dictionary. | `[9552, 28546, 278]` |

> **Common Misconception:** Developers often use "token" and "token ID" interchangeably. Strictly speaking, the **token** is the text chunk; the **token ID** is the integer representing it.

---

## 2. Vocabulary & The Reality of Special Tokens

### The Tokenizer Dictionary (Vocab)
A tokenizer contains a fixed lookup dictionary of every subword fragment it is allowed to produce, mapping each fragment to an explicit ID.

### How Special Tokens Actually Work
Tokenizers reserve special non-language tokens for structural signaling (e.g., `<bos>` for beginning-of-sequence, `<eos>` for end-of-sequence, or turn separators):

* **No Hardcoded Network Logic:** The transformer architecture does not inherently "know" what a token like ID `10` means.
* **Learned Statistical Association:** Because ID `10` was placed at the start of every single prompt across trillions of training tokens, the network learned the statistical pattern that ID `10` signals the beginning of context.
* **Predictive Orientation:** Supplying the correct special token orients the mathematical probabilities, allowing the model to generate coherent continuations rather than degrading into unstructured text.

---

## 3. Tokenizer Architecture: Model Coupling

Tokenizers are **strictly coupled** to their specific model weights:

| Model Family | Tokenizer Coupling | Behavior |
| :--- | :--- | :--- |
| **Meta Llama** | Llama Tokenizer | Uses byte-level Byte-Pair Encoding (BPE) tuned for Llama vocabularies. |
| **Microsoft Phi** | Phi Tokenizer | Specialized subword splits optimized for compact synthetic/textbook datasets. |
| **DeepSeek** | DeepSeek Tokenizer | Custom vocabulary adapted for multi-lingual and code performance. |
| **Qwen / Qwen Coder** | Qwen Tokenizer | Large vocabulary optimized for dense multilingual compression and code syntax. |

### The Token Efficiency "Red Herring"
* Some tokenizers compress sentences into fewer tokens than others.
* While fewer tokens slightly reduce input compute or per-token API costs, small differences in compression efficiency are negligible.
* **Priority Rule:** Select a model based on reasoning capability and output quality, not because its tokenizer produces 5% fewer tokens on an input prompt.

---

## 4. Token IDs vs. Vector Embeddings

Do not confuse token IDs with vector embeddings:

| Stage | Data Representation | Dimensionality | Location in Architecture |
| :--- | :--- | :--- | :--- |
| **Token ID** | Discrete integer scalar (e.g., `1842`) | 1D scalar | Model Input Layer (Lookup Key) |
| **Embedding Vector** | Continuous floating-point coordinates (e.g., `[0.12, -0.45, 0.88, ...]`) | High-dimensional dense vector (e.g., 4096-dim) | Deep within neural layers (Embedding Layer onward) |

* Statistical neural networks only accept numbers. Token IDs serve as the discrete indices used to look up high-dimensional vector embeddings inside the model's first weight matrix.

---

## 5. Transitioning Under the Hood

* **Week 3, Days 1–2:** Focused on high-level pipelines (`pipeline(task="...", model="...")`), where tokenization and decoding happen invisibly.
* **Week 3, Day 3+:** Manual execution — instantiating the standalone `AutoTokenizer`, inspecting vocabulary mappings, formatting chat templates, and passing raw tensor IDs into PyTorch models.
