# Evaluation of Portuguese Language Models

This repository presents ongoing research on fine-tuning Transformer models for Portuguese natural language understanding tasks.

## Summary of the Fine-Tuning procedure

* Step 1. Hyperparameter optimization is performed using quasi-random search based on Google's Deep Learning Playbook instructions. The best learning rate, weight decay, and adam beta1 parameters for each Transformer model on each task are identified.

* Step 2. The best hyperparameters from step 1 are used to fine-tune each model 40 times with different random seeds for up to one epoch. The 10 best models after the first epoch are selected for the next step.

* Step 3. The top 10 models from step 2 are fine-tuned for 20 epochs, generating predictions for the test set. We select the model that is closest to the average of predictions (for regression tasks) or the mode of predictions (for classification tasks). This final model is then submitted for evaluation and uploaded to the Hugging Face Hub.

## Results  

Our fine-tuning procedure has achieved results that are either slightly superior or at the same level as the previous state-of-the-art (if any).
Below is a summary of the results achieved on each dataset.

### ASSIN 2 - STS ( Semantic Textual Similarity )

| Model                                                    | Pearson | MSE  |
|----------------------------------------------------------|---------|------|
| **ruanchaves/bert-large-portuguese-cased-assin2-similarity** | **0.86**    | **0.48** |
| Previous SOTA ( for Pearson )                            | 0.852   | 0.50 |
| SOTA ( for MSE )                                | 0.817   | **0.47** |
| ruanchaves/mdeberta-v3-base-assin2-similarity            | 0.847   | 0.62 |
| ruanchaves/bert-base-portuguese-cased-assin2-similarity  | 0.843   | 0.54 |

### ASSIN 2 - RTE ( Recognizing Textual Entailment )

| Model                                                    | Accuracy | F1  |
|----------------------------------------------------------|---------|------|
| **ruanchaves/bert-large-portuguese-cased-assin2-entailment** | **0.90**    | **0.90** |
| ruanchaves/mdeberta-v3-base-assin2-entailment           | **0.90**   | **0.90** |
| Previous SOTA                           | **0.90**  | **0.90** |
| ruanchaves/bert-base-portuguese-cased-assin2-entailment | 0.88   | 0.88 |

### ASSIN - STS ( Semantic Textual Similarity )

| Model                                                    | Pearson | MSE  |
|----------------------------------------------------------|---------|------|
| ruanchaves/bert-large-portuguese-cased-assin-similarity | 0.859    | 0.3 |
| ruanchaves/mdeberta-v3-base-assin-similarity            | 0.855  | 0.39 |
| ruanchaves/bert-base-portuguese-cased-assin-similarity  | 0.847   | 0.33 |

### ASSIN - RTE ( Recognizing Textual Entailment )

| Model                                                    | Accuracy | F1  |
|----------------------------------------------------------|---------|------|
| ruanchaves/mdeberta-v3-base-assin-entailment           | 0.927   | 0.862 |
| ruanchaves/bert-large-portuguese-cased-assin-entailment | 0.92    | 0.828 |
| ruanchaves/bert-base-portuguese-cased-assin-entailment | 0.92   | 0.827 |

### HateBR ( Offensive Language Detection )


| Model                                                    | Accuracy | F1  |
|----------------------------------------------------------|---------|------|
| ruanchaves/bert-large-portuguese-cased-hatebr | 0.928    | 0.928 |
| ruanchaves/mdeberta-v3-base-hatebr          | 0.916 | 0.916 |
| ruanchaves/bert-base-portuguese-cased-hatebr | 0.914 | 0.914 |

### FaQUaD-NLI ( Question Answering )

| Model                                                    | Accuracy | F1  |
|----------------------------------------------------------|---------|------|
| ruanchaves/bert-large-portuguese-cased-faquad-nli | 0.929  | 0.93 |
| ruanchaves/mdeberta-v3-base-faquad-nli          | 0.926 | 0.926 |
| ruanchaves/bert-base-portuguese-cased-faquad-nli | 0.92 | 0.883 |

### PorSimplesSent ( Text Simplification )

| Model                                                    | Accuracy | F1  |
|----------------------------------------------------------|---------|------|
| ruanchaves/mdeberta-v3-base-porsimplessent          | 0.96 | 0.956 |
| ruanchaves/bert-base-portuguese-cased-porsimplessent | 0.942 | 0.937 |
| ruanchaves/bert-large-portuguese-cased-porsimplessent | 0.921  | 0.913 |
