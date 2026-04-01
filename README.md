# Dialogue Summarization with LoRA Fine-Tuning

A comparative study of parameter-efficient fine-tuning (PEFT) applied to abstractive dialogue summarization across three transformer architectures — evaluating both encoder-decoder and decoder-only models on the [`knkarthick/dialogsum`](https://huggingface.co/datasets/knkarthick/dialogsum) dataset.

---

## Overview

This project fine-tunes three models using **LoRA (Low-Rank Adaptation)** to explore how PEFT scales across model sizes and architectural families. Rather than treating this as a single-model tutorial, the design isolates two variables:

- **Scale:** FLAN-T5 Base (250M) vs. FLAN-T5 Large (780M) — same architecture, different capacity
- **Architecture:** FLAN-T5 Large (encoder-decoder, Seq2Seq) vs. DeepSeek-R1-Distill 1.5B (decoder-only, Causal LM)

All models are evaluated with ROUGE scores **before and after fine-tuning** against human-written summaries.

---

## Results

| Model | ROUGE-1 (Baseline) | ROUGE-1 (Fine-tuned) | Δ | ROUGE-2 (FT) | ROUGE-L (FT) | Trainable Params |
|---|---|---|---|---|---|---|
| FLAN-T5 Base | 0.216 | 0.375 | +74% | 0.128 | 0.281 | ~7M / 250M |
| FLAN-T5 Large | 0.291 | **0.421** | **+45%** | **0.152** | **0.327** | ~19M / 780M |
| DeepSeek 1.5B | 0.145 | 0.424 | +193% | 0.140 | 0.321 | ~9M / 1,770M |

> Evaluated on 100 randomly sampled examples from the 1,500-item test set using HuggingFace `evaluate` with `use_stemmer=True`. Baseline and fine-tuned models scored on the same samples for a clean before/after comparison.

**Key findings:**
- FLAN-T5 Large achieves the strongest absolute score (ROUGE-1: 0.421) from a strong non-trivial baseline (0.291), confirming that encoder-decoder architecture with instruction tuning is well-suited for summarization.
- DeepSeek's large relative gain (+193%) reflects a very low starting point — as a reasoning-distilled model, it generates verbose chain-of-thought outputs by default, which LoRA successfully overrides.
- LoRA reduces trainable parameters by **97–99% vs. full fine-tuning** across all three models, with no catastrophic forgetting of pre-trained capabilities.

---

## Models

### FLAN-T5 Base (250M parameters)
A lightweight encoder-decoder model, instruction-tuned for a wide range of NLP tasks. Served as the pipeline validation baseline. LoRA reduces trainable parameters to ~7M (~2.8% of total).

### FLAN-T5 Large (780M parameters)
Same architecture, larger capacity. Same LoRA configuration results in ~19M trainable parameters (~2.4%). Achieves the best balance of absolute ROUGE score and interpretable gain over a strong baseline.

### DeepSeek-R1-Distill-Qwen-1.5B (1.77B parameters)
A decoder-only model distilled from a reasoning-focused pre-training objective. Unlike T5, it frames summarization as a text continuation task. Included to test whether LoRA can redirect a model from reasoning-style outputs toward structured, concise summarization. LoRA reduces trainable parameters to ~9M (~0.5% of total).

---

## Technical Approach

### LoRA (Low-Rank Adaptation)

LoRA inserts small trainable matrices into the attention layers of the frozen base model. Instead of learning a full weight update `ΔW ∈ R^{d×k}`, it learns two smaller matrices `A ∈ R^{d×r}` and `B ∈ R^{r×k}` where `r << d`, with the update `ΔW = BA`. Only these matrices are trained; the base model weights remain frozen.

This approach:
- Reduces trainable parameters by 97–99% vs. full fine-tuning
- Prevents catastrophic forgetting by constraining updates to a low-rank subspace
- Enables fine-tuning large models on a single GPU with bfloat16 precision

### Data-Driven Tokenization

Rather than guessing a `max_length`, this project takes a principled approach:
1. Constructs real inference prompts (with the actual prompt template, not raw text)
2. Tokenizes without padding or truncation
3. Computes the **95th percentile** of token lengths across the dataset
4. Uses that value as the `max_length` cap

This ensures most inputs are fully represented without unnecessary padding, improving token efficiency. ROUGE scores remained stable or slightly improved with the tighter length bounds.

**Resulting max lengths:**
- T5 dialogue input: 512 tokens
- T5 summary target: 128 tokens
- DeepSeek combined prompt: 512 tokens

### Prompt Formatting

**FLAN-T5 (Seq2Seq — input and target fed separately):**
```
Summarize the following conversation.

{dialogue}

Summary:
```

**DeepSeek (Causal LM — input and target concatenated):**
```
{dialogue}

Summary: {summary}
```

### Precision

All models loaded with `torch_dtype=torch.bfloat16` to reduce GPU memory footprint. This is distinct from PyTorch AMP (`autocast`) — bfloat16 loading is a simpler, stable approach for modern GPUs with native bfloat16 support.

---

## Dataset

[`knkarthick/dialogsum`](https://huggingface.co/datasets/knkarthick/dialogsum) — 10,000+ multi-turn dialogues with manually written abstractive summaries and topic labels.

| Split | Size |
|---|---|
| Train | 12,460 |
| Validation | 500 |
| Test | 1,500 |

---

## Notebooks

Each model is documented in a self-contained notebook covering data loading, tokenization, LoRA configuration, training, and ROUGE evaluation:

| Notebook | Model |
|---|---|
| `Fine_tuning_FlanT5_base_LoRA_knkarthick_dialogsum.ipynb` | FLAN-T5 Base |
| `Fine_tuning_FlanT5_large_LoRA_knkarthick_dialogsum.ipynb` | FLAN-T5 Large |
| `Fine_tuning_DeepSeek_1.5B_LoRA_knkarthick_dialogsum.ipynb` | DeepSeek-R1-Distill 1.5B |

---

## Limitations

- ROUGE scores are computed on 100 samples from the 1,500-item test set — sufficient for directional comparison but not for high-precision absolute benchmarking.
- Models were trained for a limited number of epochs; additional training would likely improve ROUGE scores further.
- The DeepSeek training loop does not mask input tokens — loss is computed over both the dialogue and the summary. Masking the dialogue tokens (computing loss only on the summary) would be a more principled fine-tuning objective for decoder-only models.
- No semantic evaluation metrics (e.g., BERTScore) were included; ROUGE measures lexical overlap only.

---

## Dependencies

```
transformers
peft
datasets
evaluate
torch
pandas
numpy
matplotlib
```

---

## Potential Extensions

- Evaluate on the full 1,500-sample test set for more reliable ROUGE estimates
- Add BERTScore alongside ROUGE for semantic quality measurement
- Implement input token masking in the DeepSeek training loop
- Add Pegasus or BART as purpose-built summarization baselines
- Experiment with QLoRA (4-bit quantization) for larger models
- Merge LoRA weights and benchmark inference latency before vs. after fine-tuning
