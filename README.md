# NeuroGuard - AI-Powered Stroke Prediction System

NeuroGuard is a comprehensive full-stack application designed to predict the likelihood of a stroke based on patient health data. It leverages advanced machine learning models on the backend and provides a modern, responsive user interface on the frontend.

## 🚀 Project Overview

The system consists of two main components:
1.  **Backend (Python/FastAPI):** A robust API that serves a pre-trained machine learning model to predict stroke probabilities. It handles data preprocessing, model inference, and dynamic risk assessment.
2.  **Frontend (React/Vite):** A user-friendly web interface that allows users to input health metrics and receive instant, easy-to-understand risk assessments and personalized insights.

## ✨ Key Features

*   **Real-time Prediction:** Instant stroke probability calculation using a trained Machine Learning model.
*   **Dynamic Risk Assessment:** Adjusts risk thresholds based on age, glucose levels, and medical history (hypertension, heart disease).
*   **Risk Factor Analysis:** Identifies and highlights specific contributors to the calculated risk (e.g., "High Glucose Levels", "History of Hypertension").
*   **Modern UI/UX:** A clean, responsive interface built with React and Framer Motion for smooth animations.
*   **Comprehensive Data Handling:** Supports various patient attributes including BMI, smoking status, work type, and residence type.

## 🛠️ Technology Stack

### Backend
*   **Language:** Python
*   **Framework:** FastAPI
*   **ML Libraries:** Scikit-learn, Pandas, NumPy, XGBoost, Imbalanced-learn
*   **Server:** Uvicorn

### Frontend
*   **Framework:** React (Vite)
*   **Styling:** CSS (Custom Design System)
*   **Routing:** React Router DOM
*   **Animations:** Framer Motion
*   **Icons:** Lucide React

## 📂 Project Structure

```
hackthon-is/
├── main.py                 # FastAPI application entry point
├── model.pkl               # Serialized trained ML model
├── train_advanced.py       # Advanced model training script
├── requirements.txt        # Python dependencies
├── web/                    # Frontend application directory
│   ├── src/
│   │   ├── pages/          # Application pages (Home, Prediction, Results)
│   │   ├── components/     # Reusable UI components
│   │   ├── App.jsx         # Main React component
│   │   └── main.jsx        # React entry point
│   └── package.json        # Frontend dependencies
└── ...                     # Other training and utility scripts
```

## ⚡ Getting Started

### Prerequisites
*   Python 3.8+
*   Node.js & npm

### 1. Backend Setup

Navigate to the project root directory:

```bash
cd hackthon-is
```

Install the required Python packages. Note: `fastapi` and `uvicorn` are required for the API but might need to be installed explicitly if not in `requirements.txt`.

```bash
pip install -r requirements.txt
pip install fastapi uvicorn
```

Start the backend server:

```bash
python main.py
```
The API will be available at `http://localhost:8000`. You can test the health check at `http://localhost:8000/`.

### 2. Frontend Setup

Open a new terminal and navigate to the `web` directory:

```bash
cd web
```

Install the dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```
The application will be accessible at `http://localhost:5173` (or the port shown in your terminal).

## 📡 API Documentation

### `POST /predict`

Accepts patient data and returns the stroke prediction.

**Request Body (JSON):**
```json
{
  "gender": "Male",
  "age": 67,
  "hypertension": 0,
  "heart_disease": 1,
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "smoking_status": "formerly smoked"
}
```

**Response (JSON):**
```json
{
  "probability": 0.45,
  "threshold": 0.25,
  "is_high_risk": true,
  "risk_factors": [
    "Advanced Age (>60)",
    "High Glucose Levels (>200)",
    "History of Heart Disease"
  ]
}
```

## 🧠 Machine Learning Model

The project includes several scripts for training and evaluating models:
*   `train_advanced.py`: Trains the primary model with advanced preprocessing.
*   `train_ensemble.py` & `train_super_ensemble.py`: Explore ensemble methods for higher accuracy.
*   `model.pkl`: The currently active model used by the API.

## 🤝 Contributing

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.
