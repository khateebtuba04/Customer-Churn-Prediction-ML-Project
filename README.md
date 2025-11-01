
# 📊 Customer Churn Prediction ML Project

### 👩‍💻 Author: **Tuba Mariyam**

---

## 🧠 Overview
This project predicts whether a customer will **churn (leave)** or **stay** with the company using **Machine Learning**.  
The dataset used is the **Telco Customer Churn Dataset**, which contains details about customers’ billing, contracts, and service usage.

This project demonstrates:
- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Model building using **Logistic Regression**  
- Model evaluation with accuracy and confusion matrix  
- Insights and conclusions based on the analysis  

---

## 🧰 Technologies Used
| Tool | Purpose |
|------|----------|
| **Python** | Programming Language |
| **Pandas** | Data Handling |
| **NumPy** | Numerical Operations |
| **Matplotlib & Seaborn** | Data Visualization |
| **Scikit-Learn (sklearn)** | Machine Learning Model |

---

## 🗂️ Dataset
The dataset used:  
**Telco Customer Churn Dataset** – available on [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)

It contains the following key features:
- Customer demographics  
- Account information (tenure, contract type, monthly charges)  
- Churn column (Yes/No)

---

## 📈 Steps in the Project
1. **Data Loading:** Import the dataset into Python.  
2. **Data Cleaning:** Convert text to numeric values, handle missing data.  
3. **Exploratory Data Analysis:** Visualize churn rates by gender, contract type, and charges.  
4. **Feature Selection:** Use key numeric columns for training the model.  
5. **Model Training:** Logistic Regression model using `scikit-learn`.  
6. **Evaluation:** Accuracy score, confusion matrix, and classification report.  
7. **Insights:** Interpret model results and draw conclusions.

---

## 🧮 Model Performance
| Metric | Result |
|---------|---------|
| **Accuracy** | 77.7% |
| **Model Used** | Logistic Regression |
| **Confusion Matrix & Report** | Displayed via `seaborn` heatmap and sklearn metrics |

---

## 📊 Visualizations
- Churn Count Distribution  
- Gender vs. Churn Rate  
- Contract Type vs. Churn  
- Confusion Matrix Heatmap  

---

## 💡 Key Insights
- Customers with **short-term contracts** are more likely to churn.  
- **Higher monthly charges** are linked to higher churn.  
- **Gender** has minimal impact on churn.  
- Around **26% of customers** have churned overall.

---

## 🚀 How to Run This Project
1. Clone this repository:
   ```bash
   git clone https://github.com/TubaMariyam/Customer-Churn-Prediction-ML-Project.git
2.Open the project in Visual Studio Code.

3.Install the required libraries:
``pip install pandas numpy matplotlib seaborn scikit-learn

4.Run the Python file:
   .python customer_churn_prediction.py
   
5.View graphs and insights in the terminal or output window.

🏁 Results

Achieved 77.7% accuracy using Logistic Regression.
This model can help businesses identify at-risk customers and improve customer retention strategies.

🌟 Future Improvements
Try Random Forest or XGBoost for better accuracy
Add more features for deeper insights
Deploy the model using Streamlit or Flask

🧩 Project Type
📈 Supervised Machine Learning Project (Binary Classification)

💬 Connect With Me
Tuba Mariyam
📧 Data Science Enthusiast & UI/UX Designer
🌐 GitHub: github.com/TubaMariyam
