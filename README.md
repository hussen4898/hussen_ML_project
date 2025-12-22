
# hussen_ML_project

# Car CO2 Emission Predictor
This project is a Machine Learning web application designed to predict CO2 emissions from vehicles based on their specifications. It utilizes a Random Forest Regression model trained on vehicle data to provide accurate emission estimates.
## 🚀 Project Overview
The **Car CO2 Emission Predictor** solves the problem of estimating environmental impact by analyzing key vehicle characteristics. It provides a simple, user-friendly interface for users to input car details and receive instant predictions.
### Technologies Used
*   **Machine Learning**: Python, Scikit-Learn, Pandas, Joblib
*   **Backend**: FastAPI, Uvicorn
*   **Frontend**: HTML5, CSS3, JavaScript
*   **Data Processing**: Pandas, OpenPyXL
## 🛠️ Prerequisites
Before you begin, ensure you have the following installed on your system:
*   **Python 3.8+**
*   **pip** (Python package installer)
## 📦 Installation & Setup
1.  **Clone the repository** (or download the source code):
    ```bash
    git clone <repository-url>
    cd web_app_CO2_emission_from_cars
    ```
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    
    # Windows
    venv\Scripts\activate
    
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies**:
    You will need the following libraries. You can install them manually:
    ```bash
    pip install fastapi uvicorn pandas scikit-learn joblib openpyxl
    ```
## 🧠 Training the Model
The project comes with a training script to generate the Machine Learning model.
1.  Ensure your dataset is located at `data/RF_shuffled_data.xlsx`.
2.  Run the training script:
    ```bash
    python backend/train_model.py
    ```
    *   This will train the Random Forest Regressor.
    *   Save the model to `backend/rf_model.joblib`.
    *   Save evaluation metrics to `backend/metrics.json`.
## 🏃‍♂️ How to Run the Application
1.  **Start the Backend API**:
    Navigate to the project root and run:
    ```bash
    python backend/main.py
    ```
    *   The API will start at `http://localhost:8000`.
    *   You can access the automatic API docs at `http://localhost:8000/docs`.
2.  **Launch the Frontend**:
    *   Open the `frontend` folder.
    *   Double-click `index.html` to open it in your web browser.
    *   Alternatively, you can serve it with a simple HTTP server:
        ```bash
        cd frontend
        python -m http.server 3000
        ```
        Then visit `http://localhost:3000`.
## 📂 Project Structure
```
├── backend/
│   ├── main.py             # FastAPI backend application
│   ├── train_model.py      # ML model training script
│   ├── rf_model.joblib     # Saved Random Forest model (generated)
│   └── metrics.json        # Model performance metrics (generated)
├── data/
│   └── RF_shuffled_data.xlsx # Dataset (excluded from repo)
├── frontend/
│   ├── index.html          # User interface
│   ├── script.js           # Frontend logic
│   └── styles.css          # Styling
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```
