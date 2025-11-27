
# **SOCIALProject: Dual NLP Analytics Platform (Sentiment + Fake-News Detection)**

A lightweight prototype demonstrating applied NLP classification for social-media analytics.
This proof-of-concept aligns with the workflow described during development, focusing on:

* Text preprocessing and TF-IDF–based feature extraction
* Logistic Regression–based sentiment classification
* TF-IDF Pipeline–based fake-news detection
* Streamlit UI for real-time inference
* Bulk CSV processing and confidence visualization
* Prediction logging for audit and analysis

This project brings together:

* Classical ML NLP pipelines
* Human-interpretable probability outputs
* Structured datasets (Twitter sentiment + Fake/Real news)
* A unified analytics dashboard built with Streamlit

---

## **Project Brief**

Social platforms generate massive amounts of unstructured text, making manual monitoring impossible.
This prototype provides a practical demonstration of:

### **1. Sentiment Classification**

Classifies user text as:

* **Positive**
* **Neutral**
* **Negative**

Leveraging TF-IDF + Logistic Regression.

### **2. Fake-News Detection**

Detects misinformation using a trained TF-IDF pipeline on merged:

* **fake.csv** (label=1)
* **true.csv** (label=0)**

### **3. Confidence-Driven Interpretation**

For every prediction, the UI shows probability bars, helping users judge certainty.

### **4. Bulk Analytics**

Upload CSV files containing a `text` column to perform large-scale inference.

### **5. Logging & Diagnostics**

All predictions are logged for future review or offline evaluation.

This PoC demonstrates the end-to-end workflow of a simple yet functional social-text analytics system.

---

## **Project Folder Structure**

```
social-x/
├── app.py                         # Streamlit application
│
├── artifacts/                     # Saved ML artifacts + logs
│   ├── sentiment_model.joblib
│   ├── vectorizer.joblib
│   ├── fake_model_skl_pipeline.joblib
│   └── predictions_log.csv
│
├── src/
│   ├── core.py                    # Text cleaning, logging, artifact loading
│   └── __init__.py
│
├── data/
│   ├── twitter_training.csv       # Sentiment dataset
│   ├── fake.csv                   # Fake news dataset
│   └── true.csv                   # Real news dataset
│
├── notebooks/
│   └── training_notebook.ipynb    # Model training workflow
│
└── README.md                      # (this document)
```

---

## **Role of Each Component**

### **app.py**

Streamlit UI hosting both features:

* Sentiment analysis
* Fake-news classification
  Supports:
* Real-time prediction
* Probability visualization
* Bulk CSV inference
* Audit logging

### **src/core.py**

Utility layer including:

* Text cleaning
* Prediction logging
* Artifact loading

### **artifacts/**

Contains all trained models used by the app.

### **data/**

Original datasets used for training/evaluation.

### **notebooks/**

Jupyter Notebook where models were trained and exported.

---

## **Setup & Run Instructions**

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2. Ensure Artifacts Exist**

Populate `artifacts/` with:

* `sentiment_model.joblib`
* `vectorizer.joblib`
* `fake_model_skl_pipeline.joblib`

(These are produced by your training notebook.)

### **3. Launch the Streamlit App**

```bash
streamlit run app.py
```

Your browser will open the analytics dashboard.

---

## **How the System Works (High-Level)**

### **1. Data Cleaning**

Removes:

* URLs
* Mentions
* Special characters
* Excess whitespace

### **2. TF-IDF Vectorization**

Builds numerical features for:

* Sentiment classifier
* Fake-news classifier

### **3. ML Models**

* **Sentiment:** Logistic Regression
* **Fake-News:** TF-IDF Pipeline + Linear Classifier

### **4. UI Interaction**

User can:

* Enter text
* Pick preset examples
* Upload CSVs
* View confidence bars
* Review audit logs

### **5. Logging**

All predictions are recorded with:

* Timestamp
* Input text
* Model output
* Confidence scores

---

## **Evaluation Support**

When metrics are enabled, the app can:

* Load a demo dataset
* Compute classification report
* Render confusion matrices
* Show summary diagnostics

---

## **Conclusion**

This project provides a practical demonstration of applied NLP for social-data analytics using classical ML techniques.
The architecture is simple, interpretable, and extendable—ideal for student research, prototypes, and rapid experimentation.

