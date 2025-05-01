# Dialogue Summarization using T5 and Deepseek Models

This project focuses on the task of abstractive dialogue summarization using the [`knkarthick/dialogsum`](https://huggingface.co/datasets/knkarthick/dialogsum) dataset. I experimented with three different models: **Flan-T5 Base**, **Flan-T5 Large**, and **Deepseek 1.5B**, addressing challenges such as parameter-efficient fine-tuning, model scaling, and evaluation under limited resources.

##  Why These Models?

###  Flan-T5 Base (250M parameters)
I began with **Flan-T5 Base**, a lightweight and well-established model for summarization tasks. It served as a solid baseline to validate the training pipeline. To make training more efficient, I used **LoRA (Low-Rank Adaptation)**, a method that inserts small trainable matrices into existing layers while keeping most of the model frozen. This allowed me to fine-tune the model using only ~7 million parameters, making it ideal for low-resource environments without sacrificing much performance.

###  Flan-T5 Large (800M parameters)
Next, I fine-tuned **Flan-T5 Large** with the same LoRA setup to explore how scaling the model size affects performance. Despite using the same number of epochs and LoRA configuration, this model required tuning approximately 19 million parameters due to its larger architecture. The model achieved better summarization quality overall, validating the benefit of increased model capacity under the same fine-tuning conditions.

###  Deepseek 1.5B
Finally, I experimented with **Deepseek 1.5B**, a large, general-purpose language model. I applied the same LoRA configuration here as well, resulting in about 9 million trainable parameters. Unlike T5’s encoder-decoder structure, Deepseek uses a decoder-only architecture and frames summarization as a continuation task. I included this model to benchmark how a non-T5 architecture performs on this dataset. While not specifically designed for summarization, Deepseek showed promising results.

##  Tokenization and Max Length Tuning

To determine appropriate `max_length` values for each model, I used a data-driven approach. I first constructed real prompts that reflect actual inference conditions, tokenized them **without padding or truncation**, and computed the **95th percentile** of token lengths.

- For **T5 models**, which take input and target sequences separately:
  - Dialogue input length: `512`
  - Summary target length: `128`

- For **Deepseek**, which combines input and target in a single prompt:
  - Combined prompt length: `512`

This method ensured that most inputs were fully represented without unnecessary padding, improving token efficiency and potentially generalization. Although the shorter max lengths slightly increased training and validation loss, **final ROUGE scores remained stable or slightly improved**.

##  Evaluation Metrics

Model performance was assessed using the **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) metric, which is standard for summarization tasks:

- **ROUGE-1**: Measures unigram (word-level) overlap
- **ROUGE-2**: Measures bigram (two-word) overlap
- **ROUGE-L**: Measures the longest common subsequence, reflecting summary structure

For quick evaluation, I computed ROUGE scores on a **random 100-sample subset** from the 1500-item test set. Since the models were **not trained for many epochs**, their reported performance likely underrepresents their full capability. Longer training would likely improve the results.

##  Notebooks

Each model’s training, configuration, and evaluation are documented in individual notebooks:

- `Fine_tuning_FlanT5_base_model_LoRA_summerization.ipynb`
- `Fine_tuning_FlanT5_large_LoRA_knkarthick_dialogsum.ipynb`
- `Fine_tuning_DeepSeek_1.5B_LoRA_knkarthick_dialogsum.ipynb`

##  Summary

In this project, I fine-tuned and evaluated three transformer-based models for dialogue summarization using LoRA for parameter-efficient training. **Flan-T5 models** — especially the Large version — performed best overall and are well-suited for this task due to their encoder-decoder architecture. **Deepseek 1.5B**, although general-purpose and decoder-only, produced solid results and highlighted the flexibility of LoRA fine-tuning. This work demonstrates how to effectively adapt large language models for summarization tasks under resource constraints.
