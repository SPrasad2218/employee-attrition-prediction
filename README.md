# Employee Attrition Prediction & Dashboard

An end-to-end Machine Learning project that analyzes HR data to predict whether an employee is at risk of leaving the company. It features an interactive **Streamlit dashboard** built with Python to test predictions dynamically.

## 📊 Visualizations & Results
Model evaluation graphs, performance comparison matrices, and feature importance breakdowns are automatically tracked and exported into the `./outputs/` directory upon executing the pipeline.

## 🛠️ Project Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/SPrasad2218/employee-attrition-prediction.git
cd employee-attrition-prediction
```

### 2. Install dependencies
Make sure you have your virtual environment active, then run:
```bash
pip install -r requirements.txt
```

### 3. Run Model Training
To train the predictive machine learning model and save the artifact plots:
```bash
python attrition_project.py
```

### 4. Launch the Interactive Dashboard
To fire up the Streamlit interface application locally in your browser:
```bash
streamlit run app.py
```

## 🧰 Tech Stack
* **Language:** Python 3.13
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Joblib
* **Interface/App framework:** Streamlit