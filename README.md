# 🔬 Glassify – ML-Based Glass Type Classifier

A web-based Machine Learning application built with **Streamlit** that classifies types of glass based on their chemical composition. Powered by **Gradient Boosting**, this app allows both single and batch predictions with a beautiful, interactive UI enhanced by **glassmorphism** styling.

---

## 📊 Dataset

This project uses the **[UCI Glass Identification Dataset](https://www.kaggle.com/datasets/uciml/glass)**, which contains the chemical analysis of glass samples. The objective is to classify glass into one of several predefined types.

- **Source:** [Kaggle - UCI Glass Dataset](https://www.kaggle.com/datasets/uciml/glass)
- **Instances:** 214
- **Features:** 9 chemical attributes
- **Target Classes:** 6 types (excluding class 4)

| Feature | Description              |
|---------|--------------------------|
| RI      | Refractive Index         |
| Na      | Sodium (Na₂O)            |
| Mg      | Magnesium                |
| Al      | Aluminum                 |
| Si      | Silicon Dioxide          |
| K       | Potassium                |
| Ca      | Calcium                  |
| Ba      | Barium                   |
| Fe      | Iron                     |

---

## 🧠 Model Details

- **Algorithm:** Gradient Boosting Classifier  
- **Preprocessing:** SMOTE (Synthetic Minority Oversampling)  
- **Scaler:** StandardScaler  
- **Training Accuracy:** 98.54%  
- **Testing Accuracy:** 89.47%  
- **Cross-Validated Accuracy:** 88.32% ± 2.87%  
- **F1-Score:** ~0.89

---

## 📁 Project Structure


Glassify-ML-Based-Glass-Type-Classifier/
├── app.py
├── glass.csv
├── scaler.pkl
├── gradient_boosting_model.pkl
├── images.jpg
├── requirements.txt
├── GlassClassification_EDA.ipynb
├── Glassify_ML_Based_Glass_Classifier.ipynb
└── README.md


---

## 🚀 Live Demo

👉 [Click here to try the app on Streamlit](https://co4s3g27sdtvavunahvy9a.streamlit.app/)

---

## 💻 Local Installation

```bash
git clone https://github.com/adithyapurama/-Glassify-ML-Based-Glass-Type-Classifier.git
cd Glassify-ML-Based-Glass-Type-Classifier
pip install -r requirements.txt
streamlit run app.py

📦 Dependencies
streamlit

pandas

numpy

scikit-learn

plotly

imbalanced-learn

joblib

📚 Notebooks
GlassClassification_EDA.ipynb: Exploratory Data Analysis

Glassify_ML_Based_Glass_Classifier.ipynb: Model training & evaluation

🔖 Glass Type Mapping
Class	Glass Type
1	Building Windows (Float Processed)
2	Building Windows (Non-Float)
3	Vehicle Windows (Float Processed)
5	Containers
6	Tableware
7	Headlamps

👨‍💻 Author
Adithya Purama
Polu Haritha
GitHub Profile

📃 License
For educational use only. Based on publicly available data.


---