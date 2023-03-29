# Evaluation of Portuguese Language Models

This repository presents ongoing research on fine-tuning Transformer models for Portuguese natural language understanding tasks.

## Summary of the Fine-Tuning procedure

* 1. Hyperparameter optimization is performed using quasi-random search based on Google's Deep Learning Playbook instructions. The best learning rate, weight decay, and adam beta1 parameters for each Transformer model on each task are identified.

* 2. The best hyperparameters from step 1 are used to fine-tune each model 40 times with different random seeds for up to one epoch. The 10 best models after the first epoch are selected for the next step.

* 3. The top 10 models from step 2 are fine-tuned for 20 epochs, generating predictions for the test set. We select the model that is closest to the average of predictions (for regression tasks) or the mode of predictions (for classification tasks). This final model is then submitted for evaluation and uploaded to the Hugging Face Hub.

## Results  

Our fine-tuning procedure has achieved results that are either slightly superior or at the same level as the previous state-of-the-art (if any).
Below is a summary of the results achieved on each dataset.

### ASSIN 2

| Model                                                    | Pearson | MSE  |   |   |
|----------------------------------------------------------|---------|------|---|---|
| ruanchaves/bert-large-portuguese-cased-assin2-similarity | 0.86    | 0.48 |   |   |
| Previous SOTA ( for Pearson )                            | 0.852   | 0.50 |   |   |
| Previous SOTA ( for MSE )                                | 0.817   | 0.47 |   |   |
| ruanchaves/mdeberta-v3-base-assin2-similarity            | 0.847   | 0.62 |   |   |
| ruanchaves/bert-base-portuguese-cased-assin2-similarity  | 0.843   | 0.54 |   |   |