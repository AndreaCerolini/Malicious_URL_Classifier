# Malicious URL Classifier

This repository includes all the files used to train and test a machine learning model that classifies URLs into four categories: **benign**, **defacement**, **malware**, and **phishing**.

## Repository Structure

- **`history_of_the_classifier.ipynb`**  
  A Jupyter notebook that shows the full development of the classifier, from the first version to the final one with comments that explain every step.

- **`Malicious_URL_Classifier.py`**  
  The main script that runs the training process.  
  It loads the preprocessed data, trains the model, uses early stopping, and saves the best model at the end.

- **`models/best_model_embed.pth`**  
  The best model saved during training.  
  This file contains:
  - `epoch`
  - `model_state`
  - `optimizer_state`
  - `val_loss`

## Requirements

To install all the required libraries, run:

```bash
pip install -r requirements.txt
