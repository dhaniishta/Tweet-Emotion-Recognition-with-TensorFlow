# Tweet-Emotion-Recognition-with-TensorFlow

This project explores emotion recognition in tweets using deep learning. Built using **TensorFlow**, it leverages a labeled dataset of social media text to train a model that classifies emotions like **joy, anger, sadness, fear**, and more.

> **Goal:** Predict the emotional tone behind a tweet using an end-to-end NLP pipeline and visualize insights from the data.

---

##  Dataset

- [Emotion Dataset](https://github.com/dair-ai/emotion_dataset) by [dair-ai](https://github.com/dair-ai)  
- Consists of tweets labeled with 6 emotion categories.

---

## Pipeline Overview

### Task Breakdown

1. **Data Loading & Exploration**  
   - Loaded train/validation/test splits using HuggingFace's `datasets` library  
   - Explored class distribution and tweet lengths

2. **Text Preprocessing**  
   - Tokenized tweets using Keras `Tokenizer`  
   - Padded sequences to ensure uniform input length

3. **Label Encoding**  
   - Mapped emotion classes to numeric IDs for model training

4. **Model Architecture**
   - `Embedding` layer to convert words to vector representations
   - `Bidirectional LSTM` layers to capture tweet context
   - `Dense (Softmax)` output layer for emotion classification  
   - Total output classes: 6 (one per emotion)

5. **Training & Evaluation**  
   - Trained for 20 epochs with early stopping on validation accuracy  
   - Visualized training history and confusion matrix  
   - Final evaluation on test data with per-tweet prediction

---

## Key Highlights

| Feature                  | Details |
|--------------------------|---------|
| **Model Type**           | Bi-LSTM (Bidirectional LSTM) |
| **Framework**            | TensorFlow / Keras |
| **Embedding Size**       | 16 |
| **Max Tweet Length**     | 50 tokens |
| **Training Accuracy**    | ~88–90% |
| **Evaluation Metrics**   | Accuracy, Confusion Matrix |
| **Dataset Size**         | ~20k tweets |
| **Label Categories**     | Joy, Sadness, Anger, Fear, Surprise, Love |

---

## Try It Yourself

You can test this notebook interactively using:
- Jupyter Notebook / Colab
- Clone this repo and run:  
  ```bash
  pip install tensorflow datasets matplotlib
