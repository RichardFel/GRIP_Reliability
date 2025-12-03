# GRIP_Human_Activity_Patterns
Machine-learning pipelines to classify activity intensities in people with Chronic Pain

---

## 📖 Overview

The **GRIP project** (beweeGsensoren voor mensen met chRonIsche Pijn) aims to develop robust machine-learning models that classify **real-world activity intensities** using wrist-worn accelerometers in people with Chronic Pain (CP).

Because activity behaviour in CP differs from healthy individuals, and because labeled data is limited, this repository provides a complete, reproducible pipeline for:

- Processing **lab-based** annotated accelerometer data  
- Processing **real-world capture** data with researcher-annotated activity logs  
- Training and evaluating multiple machine-learning models  
- Comparing CP vs. healthy patterns  
- Training a **final production-ready model** using all available data  
- Validating the model on **criterion validity** real-world data  

This repository contains all scripts necessary to reproduce the results for the referenced publication.

---

## 📝 How to Cite

If you use this software, please cite:

**Annet Doomen, Richard A. W. Felius**  
*An open-source deep learning model to classify real-world activity intensities in persons with chronic pain using a wrist-worn accelerometer.*

---

## 📂 Data Sources

### **1. Lab-based annotated data**
Participants (both healthy and CP) performed a series of structured activities at varying movement intensities.  
Researchers labeled each activity segment so accelerometer data could be aligned with the correct activity intensity.

### **2. Real-world “capture” data**
Participants wore an accelerometer for **four hours** during daily life while a researcher continuously annotated their activities.  
These labels were manually converted into **MET-scores**, which served as criterion validity outcomes.

### **3. Criterion validity dataset**
Used to evaluate how well the final model generalizes to unconstrained real-world behaviour.



## 🧠 Pipeline Summary

### **Data Processing**

The pipeline:

- Loads raw accelerometer data  
- Aligns movement segments with annotation labels  
- Applies preprocessing & filtering  
- Windows the data  
- Extracts features or prepares raw signal windows  
- Exports standardized datasets for training/testing  
---

### **Model Evaluation (evaluate_models.py)**

This script:

- Trains multiple ML architectures  
- Evaluates accuracy, F1, precision, recall  
- Generates confusion matrices  
- Selects the best-performing model  


---

### **Final Model (definitive_model.py)**

The selected model is retrained and evaluated separately for:

- Lab CP  
- Lab healthy  
- Real-world CP  
- Real-world healthy  

Multiple repetitions are used to assess generalization stability.

---

### **Production Model (production_model.py)**

A final model is trained using **all available data** and then evaluated on the **criterion validity** dataset.  
This assesses real-world performance under deployment-like conditions.

**TODO:** Add final hyperparameters and architecture summary.

---

## 📊 Outputs

The repository generates:

- Confusion matrices  
- Performance tables  
- Model comparison figures  
- Group-specific evaluation outcomes  
- Exported trained models  

---

## ▶️ How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Process the data
```
python data_processing/process_lab_data.py
python data_processing/process_capture_data.py
```

### 3. Train and evaluate models
```
python model_training/evaluate_models.py
```

### 4. Train final & production models
```
python model_training/definitive_model.py
python model_training/production_model.py
```

---

## 📦 Requirements

- Python 3.9+  
- TensorFlow / Keras  
- NumPy & Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  

---

## 👩‍🔬 Contributors

- **Richard A. W. Felius**  

---
