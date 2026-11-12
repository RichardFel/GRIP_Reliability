# GRIP_Human_Activity_Patterns
Reliability of Activity Patterns measured with Accelerometers in Chronic Pain Patients
test
---

## 📖 Overview

The **GRIP project** (beweeGsensoren voor mensen met chRonIsche Pijn) aims to develop robust machine-learning models that classify **real-world activity intensities** using wrist-worn accelerometers in people with Chronic Pain (CP).

Because activity behaviour in CP differs from healthy individuals, and because labeled data is limited, this repository provides a complete, reproducible pipeline for:

- Processing accelerometer data with a trained model
- Transfrom activity patterns into characteristics
- Determine the ICC values per day

This repository contains all scripts necessary to reproduce the results for the referenced publication.

---

## 📝 How to Cite

If you use this software, please cite:

**Annet Doomen, Richard A. W. Felius**  
*Reliability of Activity Patterns measured with Accelerometers in Chronic Pain Patients.*

---

## 📂 Data Sources

### **1. Real-world data**
Participants (both healthy and CP) wore the sensor for up to 14 days.

## 🧠 Pipeline Summary

### **Data Processing**

The pipeline:

- Loads raw accelerometer data  
- Transform into characteristics
- Calculate ICC  
---


## 📊 Outputs

The repository generates:

- Excel of the outcomes

---

## ▶️ How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Process the data
```
python main.py
```


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
