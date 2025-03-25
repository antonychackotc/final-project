# final-project

# Fraud Detection and NLP-based Text Summarization - README

## Project Overview
This repository contains my work on two major projects:
1. **Insurance Fraud Detection** - Predicting fraudulent insurance claims using machine learning models.
2. **NLP-based Sentiment Analysis & Text Summarization** - Sentiment classification using BERT embeddings and multi-language summarization using mBART/mT5.

## Timeline & Learning Journey
### Started: February 18, 2025 | Last Updated: March 25, 2025

### **1. Insurance Fraud Detection**
- Built a fraud risk prediction app using Streamlit and TensorFlow/Keras.
- Used Random Forest initially but faced **overfitting** (100% training accuracy, 96.5% test accuracy).
- Used **synthetic data** but realized poor accuracy due to randomness.
- **Feature Engineering & Scaling:** Applied MinMax scaling and used `scaler.pkl` for consistent preprocessing.
- **Hyperparameter Tuning & Model Evaluation:**
  - Used cross-validation, confusion matrix, precision, recall, F1-score, cross-entropy loss, and MSE.
  - Applied gradient descent for MSE reduction and L1/L2 regularization for optimization.
  
**Key Takeaways:**
- **Synthetic datasets aren't ideal for ML**—real-world data improves accuracy.
- Machine learning often outperforms deep learning in tabular datasets unless the dataset is very large.

---
### **2. NLP-based Sentiment Analysis & Multi-language Summarization**
#### **Sentiment Analysis**
- Used **BERT embeddings** with Random Forest, XGBoost, MLP, CNN+LSTM, but results were poor due to synthetic data.
- Switched to **real Twitter data**, leading to better predictions and accuracy.

#### **Multi-language Text Summarization**
- Fine-tuned **mBART & mT5** for multilingual summarization.
- **mBART outperformed mT5**, but requires a **GPU**.
- Due to system limitations, switched to **BART-Small**.
- Real-world data increased **ROUGE & BLEU scores**, improving translation fluency & summarization quality.

**Key Takeaways:**
- **BERT is superior to TF-IDF & Word2Vec** due to pretraining.
- **Deep learning isn't always better than ML**—ML is preferable when deep learning struggles with gradient/loss control.
- **Real-time data drastically improves NLP model performance.**

## Final Learnings
- **Synthetic data is not suitable for real-world predictions.**
- **Deep learning helps with loss reduction, but accuracy gains are limited in small datasets.**
- **Real-world data improves all ML and NLP models significantly.**
- **Pretrained models (like BERT & mBART) outperform custom-built embeddings.**

## Next Steps
- Further optimize models with real-world datasets.
- Explore different transformer architectures for NLP tasks.

---

### 🚀 **Conclusion**
This project provided hands-on experience in fraud detection, sentiment analysis, and text summarization using machine learning and deep learning. The biggest learning was the importance of real-world data for model performance. Future improvements will focus on refining preprocessing techniques and using advanced transformer models.

### 🏆 **Technologies Used**
- Python, TensorFlow/Keras, PyTorch, Streamlit
- BERT, mBART, mT5, BART-Small
- Random Forest, XGBoost, CNN, LSTM, Naive Bayes
- Feature Engineering, MinMax Scaling, Outlier Handling (Winsorization)

