# Malicious URL Classifier

This repository contains all the resources for training and evaluating a machine learning model that classifies URLs into categories such as **benign**, **defacement**, **malware**, and **phishing**.

---

## Repository Structure

- **`history_of_the_classifier.ipynb`**  
  A Jupyter notebook showing the full evolution of the classifier — from early prototypes to the final model — with detailed comments explaining each step.

- **`Malicious_URL_Classifier.py`**  
  The main training script. It loads the preprocessed dataset, trains the model, applies early stopping, and saves the best-performing model checkpoint.

- **`models/best_model_embed.pth`**  
  The best model produced during training, stored as a PyTorch checkpoint.  
  It contains:
  - `epoch`
  - `model_state`
  - `optimizer_state`
  - `val_loss`

---

## Requirements

To install all dependencies, simply run:

```bash
pip install -r requirements.txt
