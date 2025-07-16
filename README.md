# Airline-Flight-Delay-Analysis-Capstone-Project

## 👩‍💻 Author

**Dhanshree Hood**  
M.Tech Data Engineering, IIT Jodhpur  
[GitHub](https://github.com/dhanshreehood)

---

# ✈️ Airline Flight Delay Analysis – Capstone Project

This capstone project explores the patterns and causes behind flight delays using real-world airline data. By applying data analysis, visualization, and machine learning techniques, the project aims to identify key factors influencing delays and help optimize flight operations.

---

## 📌 Objective

To analyze flight delay trends and develop predictive models that can forecast whether a flight will be delayed, based on various features such as origin, destination, scheduled departure time, weather, and airline information.

---

## 📁 Dataset

- **Source**: [U.S. Department of Transportation – Bureau of Transportation Statistics](https://www.transtats.bts.gov/)
- **Contains**:
  - Flight Date, Origin & Destination
  - Carrier Information
  - Scheduled and Actual Departure/Arrival Times
  - Delay Causes (Weather, Carrier, NAS, etc.)

---

## 🧰 Tools & Technologies

- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn** (for EDA and visualizations)
- **Scikit-learn** (for modeling)
- **Jupyter Notebook**
- **Power BI** *(optional – for dashboard creation)*

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights explored:
- Top airlines with the highest delays
- Busiest airports and delay ratios
- Impact of time of day and seasons on delays
- Delay reasons (Weather, NAS, Security, etc.)

Visualizations were created to support hypothesis testing and understand patterns in delays across different dimensions.

---

## 🤖 Predictive Modeling

Developed classification models to predict if a flight will be delayed using the following algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### 🏆 Model Evaluation Metrics:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC Curve

---

## 📊 Power BI Dashboard (Optional)

An interactive Power BI dashboard was built to present:
- Flight delay trends
- Airport-wise performance
- Airline performance comparison
- Real-time slicers for date, airport, airline, etc.

---

## ✅ Conclusion

- Weather and time of day are significant contributors to delays.
- Certain airports and carriers show higher delay ratios consistently.
- Machine Learning models achieved promising accuracy in predicting delay probability.

---

## 🔗 Repository Structure

Airline-Flight-Delay-Analysis-Capstone-Project/
│
├── data/ # Raw and cleaned datasets
├── notebooks/ # Jupyter notebooks for EDA and modeling
├── visuals/ # Charts and graphs
└── README.md # Project documentation


---

## 📌 Future Enhancements

- Integrate live flight data for real-time delay prediction
- Deploy model as an API/web app using Flask or Streamlit
- Include weather API data for external feature enhancement

---
