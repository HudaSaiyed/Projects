**Twitter Sentiment Analysis**
This project implements a deep learning–based multi-class sentiment classification system using Twitter data. The model predicts sentiment categories from tweet text using a Bidirectional LSTM neural network.

**Overview**
The system classifies tweets into four sentiment categories:
- Positive
- Negative
- Neutral
- Irrelevant
The model is trained on a large Twitter dataset and applies advanced text preprocessing techniques before feeding data into a deep learning architecture.

**Dataset Information**
- Total records: 74,000+ tweets
- Columns: ID, Game, Sentiment, Text
- Sentiment classes: Positive, Negative, Neutral, Irrelevant
After cleaning, invalid entries were removed before model training.

**Data Preprocessing**
The following NLP preprocessing steps were applied:
- Text normalization (lowercasing)
- HTML tag removal
- URL removal
- Punctuation removal
- Stopword removal (NLTK)
- Stemming (Porter Stemmer)
- Alphanumeric filtering
- Tokenization using Keras Tokenizer
- Sequence padding (max length = 200)

**Model Architecture**
- Embedding Layer (10,000 vocabulary size, 128 dimensions)
- Bidirectional LSTM (32 units)
- Dropout layers for regularization
- Dense layers with L2 regularization
- Softmax output layer (4 classes)
Total parameters: ~1.3 million
EarlyStopping was used to prevent overfitting.

**Model Performance**
- Train/Test Split: 80:20
- Test Accuracy: 81.6%
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
Training and validation accuracy/loss curves were plotted to evaluate generalization.

**Tech Stack**
- Python
- Pandas
- NumPy
- NLTK
- TensorFlow
- Keras
- Matplotlib
