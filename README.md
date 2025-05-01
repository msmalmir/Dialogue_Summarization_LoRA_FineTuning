# Dialogue Summarization using T5 and Deepseek Models

This project focuses on the task of abstractive dialogue summarization using the [`knkarthick/dialogsum`](https://huggingface.co/datasets/knkarthick/dialogsum) dataset. I experimented with three different models: **Flan-T5 Base**, **Flan-T5 Large**, and **Deepseek 1.5B**, addressing a range of challenges including resource optimization, model selection, and evaluation design.

## 🔧 Why These Models?

### 🧠 Flan-T5 Base (250M parameters)
I began with **Flan-T5 Base**, a lightweight and widely used model for text generation and summarization tasks. Its smaller size made it a great starting point to test the training pipeline and observe the model's ability to learn from conversational data. To optimize training and reduce memory and compute requirements, I used **LoRA (Low-Rank Adaptation)**. LoRA is a fine-tuning technique that significantly reduces the number of trainable parameters by inserting low-rank matrices into the model's attention layers. This allowed me to fine-tune the model efficiently on consumer-grade hardware while still achieving competitive performance.

### 🚀 Flan-T5 Large (800M parameters)
After validating my pipeline with the base model, I moved on to **Flan-T5 Large**, which has over three times the parameters. The goal was to see how scaling the model size would affect performance. With its larger capacity, this model achieved better results with fewer training steps. I did not use LoRA here to allow full fine-tuning and to maximize the model’s representational power. This model demonstrated the best performance overall on the dataset within the training time constraints.

### 🔍 Deepseek 1.5B
Out of curiosity and exploration, I also experimented with **Deepseek 1.5B**, a newer and larger language model. While this model was not originally designed for summarization, I wanted to test its generalization ability on the DialogSum dataset. Unlike T5, Deepseek models do not use separate encoder-decoder architecture and instead treat summarization as a sequence continuation problem. The results were surprisingly strong, although training took significantly longer due to the model's size and structure. This model helped me benchmark a non-T5 architecture for this task and understand the trade-offs in general-purpose models versus task-specific ones.

## 🛠 Tokenization and Max Length Tuning

Selecting the right `max_length` for tokenization was a critical part of training. Initially, I set this value to `2048` for all models. However, I later refined this decision using a more data-driven approach: by constructing full prompts (as they would appear during real inference), tokenizing them without padding or truncation, and then computing statistical metrics like the 95th percentile of token lengths.

For **T5 models**, since input and output are handled separately, I set:
- Dialogue input length to `512`
- Summary target length to `128`

For **Deepseek**, where input and output are combined in a single prompt, I set:
- Combined prompt length to `512`

Although this adjustment caused a slight increase in training and validation loss, it led to more consistent input formatting and better generalization. ROUGE scores remained stable or slightly improved.

## 📊 Evaluation Metrics

I used the **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) metric to evaluate model performance. It measures the overlap between generated summaries and human-written references. Specifically:
- **ROUGE-1**: unigram overlap (word-level recall)
- **ROUGE-2**: bigram overlap (phrase-level fluency)
- **ROUGE-L**: longest common subsequence (summary structure)

Due to time constraints, I limited evaluation to **100 samples randomly selected** from the 1500-item test set. Additionally, models were **not trained for many epochs**, so the reported results likely underestimate their full potential. Extended training could yield further improvements.

## 📁 Notebooks

Each training experiment is documented in its corresponding Jupyter notebook, with code, logs, and evaluation results:

- `Fine_tuning_FlanT5_base_model_LoRA_summerization.ipynb`
- `Fine_tuning_FlanT5_large.ipynb`
- `Fine_tuning_Deepseek_1.5B.ipynb`

## 📝 Summary

In this project, I explored three architectures to fine-tune and evaluate models for dialogue summarization. T5 models, especially **Flan-T5 Large**, performed best overall and are well-suited to summarization tasks due to their encoder-decoder structure. **Deepseek 1.5B**, while not task-specific, showed promising results and served as a useful benchmark. Through this work, I also addressed challenges like efficient fine-tuning using LoRA, optimal token length selection, and evaluation with limited computational resources.
