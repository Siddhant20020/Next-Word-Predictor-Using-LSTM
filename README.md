# Next Word Predictor Using LSTM

This project implements a **next-word prediction model** using LSTM networks. The system predicts the most probable next word in a sentence based on a given text input.

## Features

- Built an **LSTM model** for next-word prediction using the Kaggle **Quotes dataset** (30k quotes).
- Implemented **tokenization, padding, and sequence generation** for text preprocessing.
- Model architecture:
  - **Embedding layer** to convert words to dense vectors.
  - **2 LSTM layers** for sequential modeling.
  - **Dense softmax output** for predicting probabilities of the next word.
- Achieved:
  - **Top-1 accuracy:** 35%  
  - **Top-3 accuracy:** 60%
- Deployed as a **web demo** for real-time next-word prediction.

## Dataset

- **Kaggle Quotes Dataset**: Contains 3039 quotes.
- Used for training the sequence prediction model.


