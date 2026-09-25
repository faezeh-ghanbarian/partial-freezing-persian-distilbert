# Partial Freezing of Multilingual DistilBERT for Persian POS Tagging

**Author:** Faezeh Ghanbarian  
**Challenge:** Graduate Candidate Challenge - Part 1  
**Task:** Partial Freezing of MLMs for POS Tagging

## Project overview

This project investigates the effect of partial layer freezing when fine-tuning a distilled multilingual Transformer for Persian part-of-speech (POS) tagging. It compares full fine-tuning with a configuration in which the embedding module and the first three Transformer layers are frozen.

The experiment evaluates whether the number of updated parameters and observed training time can be reduced while retaining most of the validation accuracy achieved by full fine-tuning.

## Dataset

The experiment uses the Persian PerDT treebank from Universal Dependencies ('UD_Persian-PerDT').

| Split | Original size | Subset used | Purpose |
|---|---:|---:|---|
| Training | 26,196 sentences | 8,000 sentences | Model training |
| Development | 1,456 sentences | 1,000 sentences | Reported evaluation |
| Test | 1,455 sentences | Not evaluated | Held out |

The training and development subsets were sampled after independently shuffling each split with seed 42. The official test split was downloaded and preprocessed but was not used for the reported results.

## Preprocessing

The model tokenizer produces subword pieces, whereas the treebank provides one UPOS label per word. To align the labels:

- sentences are tokenized with 'is_split_into_words=True';
- sequences are truncated to a maximum length of 128 tokens;
- the POS label is assigned only to the first subword of each word;
- special tokens and subsequent subwords receive the ignore index '-100';
- accuracy is calculated only over positions whose labels are not '-100'.

## Model

The experiments use:

'''text
distilbert-base-multilingual-cased
'''

The checkpoint contains six Transformer layers and is loaded with a newly initialized token-classification head.

## Experimental configurations

### 1. Full fine-tuning

All pretrained model parameters and the token-classification head remain trainable.

### 2. Partial freezing

A new instance of the same checkpoint is created. The following components are frozen:

- the embedding module;
- Transformer layer 0;
- Transformer layer 1;
- Transformer layer 2.

Transformer layers 3-5 and the token-classification head remain trainable. Parameters in the frozen components are assigned 'requires_grad=False', so only the unfrozen parameters are updated.

## Shared hyperparameters

| Setting | Value |
|---|---:|
| Learning rate | 2e-5 |
| Training batch size | 8 |
| Evaluation batch size | 8 |
| Epochs | 3 |
| Weight decay | 0.01 |
| Maximum sequence length | 128 |
| Training seed | 42 |
| Data seed | 42 |
| Primary metric | First-subword development accuracy |

Both configurations use the same sampled training and development datasets, tokenizer, preprocessing function, data collator, metric, and primary hyperparameters.

## Results

The values below are taken from the epoch-3 Hugging Face Trainer logs stored in the final T4 GPU execution of the notebook.

| Configuration | Trainable parameters | Trainable ratio | Development accuracy | Trainer runtime |
|---|---:|---:|---:|---:|
| Full fine-tuning | 134,747,153 | 100.00% | 95.8717% | 233.9971 s |
| Partial freezing | 21,276,689 | 15.79% | 94.2419% | 127.2452 s |
| Observed change | -113,470,464 | -84.21 percentage points | -1.6298 percentage points | -45.6210% |

### Epoch-level validation results

| Epoch | Full validation loss | Full accuracy | Frozen validation loss | Frozen accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.160743 | 94.8211% | 0.216163 | 92.8620% |
| 2 | 0.139302 | 95.5650% | 0.182253 | 94.0034% |
| 3 | 0.135400 | 95.8717% | 0.173906 | 94.2419% |

Partial freezing retained most of the full fine-tuning accuracy while reducing the trainable-parameter share to 15.79%. In this run, the Trainer runtime decreased by approximately 45.6210%, while development accuracy decreased by 1.6298 percentage points.

Because each configuration was run once, these differences should be treated as preliminary observed results rather than statistically reliable estimates.

## Time spent

The complete GPU-backed notebook run took approximately 20 minutes (about 0.33 hours) end to end, including downloads, preprocessing, model setup, training, evaluation, and review.

Within that run, the Hugging Face Trainer recorded:

- full fine-tuning: 233.9971 seconds;
- partial freezing: 127.2452 seconds.

## Files

- 'Copy_of_Faezeh_Ghanbarian_Partial_Freezing_MLM_POS_Tagging_SUBMISSION.ipynb'  
  Jupyter/Colab notebook containing dataset download, preprocessing, word-to-subword label alignment, model training, partial freezing, evaluation, parameter counting, and stored outputs.

- 'Investigating_Partial_Freezing_Multilingual_DistilBERT_Persian_POS_Tagging.pdf'  
  Brief report describing the experimental design, verified results, interpretation, limitations, time spent, and possible research extensions.

- 'README_FINAL.md'  
  Project summary and reproduction guidance.

## Reproduction

1. Open the notebook in Google Colab.
2. Select a T4 GPU runtime if available.
3. Run the cells in order.
4. Allow the notebook to download the Universal Dependencies files and required Python packages.
5. Inspect the epoch-level Trainer tables for the authoritative validation results.

Running the complete notebook retrains both models. Runtime and accuracy may vary slightly across environments and executions because the experiment contains stochastic components and was not repeated across multiple seeds.

## Limitations and extensions

The experiment evaluates one language, one model, one subset size, one freezing boundary, and one run per configuration. Future work could evaluate multiple random seeds, alternative freezing boundaries, per-POS-tag performance, peak memory, energy consumption, LoRA under matched budgets, adaptive layer selection, and typologically diverse languages.

The raw UPOS inventory also contains an underscore value present in the parsed source data. A stricter future preprocessing pipeline should inspect these records and determine whether missing or nonstandard UPOS entries should be excluded before label-inventory construction.
