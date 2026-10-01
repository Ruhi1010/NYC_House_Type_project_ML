# 🗽 NYC Room Type Prediction 


## 🚀 Live Demo

**Project:** https://nyc-house-type-project-ml-1.onrender.com

A complete machine learning application for predicting the **room type of a New York City Airbnb listing** from listing characteristics such as location, price, minimum nights, reviews, availability, and neighbourhood.

The project combines a **scikit-learn machine learning pipeline**, a **FastAPI REST API**, and a responsive **HTML/CSS/JavaScript frontend**. The application is designed to be deployed as a web service, with the FastAPI backend hosted on platforms such as Render.

---

## 📌 Project Overview

The goal of this project is to build a supervised machine learning classification system that predicts one of three Airbnb room types:

- **Entire home/apt**
- **Private room**
- **Shared room**

The model is trained using the **New York City Airbnb Open Data (AB_NYC_2019)** dataset.

The final trained preprocessing and classification pipeline is stored in `Model_Pipeline.pkl` and is loaded by the FastAPI backend to make predictions through the `/predict` endpoint.

---

## ✨ Features

- 🏠 Predicts Airbnb room type from listing information
- 🤖 Machine learning classification using scikit-learn
- 🧹 Automated preprocessing through a reusable pipeline
- 📊 Numerical feature transformation and scaling
- 🔤 Categorical feature encoding
- 🌲 Random Forest classification
- ⚖️ Class-balanced training using `class_weight='balanced'`
- 🔎 Hyperparameter optimization using `RandomizedSearchCV`
- 🌐 REST API built with FastAPI
- 🖥️ Interactive frontend using HTML, CSS, and JavaScript
- 📈 Displays prediction probabilities for each room type
- ❤️ API connection/health status indicator
- ☁️ Ready for cloud deployment, including Render
- 🔐 CORS enabled for communication between frontend and API

---

## 🧠 Machine Learning Workflow

The project follows this general workflow:

```text
Raw NYC Airbnb Dataset
        ↓
Exploratory Data Analysis
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
Preprocessing Pipeline
        ↓
Try Multiple Classification Algorithms
        ↓
Random Forest Selection
        ↓
Hyperparameter Tuning
        ↓
Final Test Evaluation
        ↓
Save Complete Pipeline
        ↓
FastAPI REST API
        ↓
Web Frontend
        ↓
Room Type Prediction
```

---

## 📊 Dataset

The project uses the **New York City Airbnb Open Data (AB_NYC_2019)** dataset.

Dataset source:

[Kaggle — New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)

The project also includes a local copy of the dataset:

```text
AB_NYC_2019.csv
```

### Dataset size

The included dataset contains:

- **48,895 rows**
- **16 original columns**

### Target distribution

| Room Type | Number of Listings |
|---|---:|
| Entire home/apt | 25,409 |
| Private room | 22,326 |
| Shared room | 1,160 |

The target is therefore somewhat imbalanced, particularly for the `Shared room` class. The model-training process accounts for this by using class-balanced classification where appropriate and by evaluating models using **macro F1** in addition to accuracy.

---

## 🧾 Input Features

The final prediction API uses the following 10 features:

| Feature | Type | Description |
|---|---|---|
| `latitude` | Float | Listing latitude |
| `longitude` | Float | Listing longitude |
| `price` | Float | Price per night |
| `minimum_nights` | Integer | Minimum number of nights required |
| `number_of_reviews` | Integer | Total number of reviews |
| `reviews_per_month` | Float | Average reviews per month |
| `calculated_host_listings_count` | Integer | Number of listings managed by the host |
| `availability_365` | Integer | Number of available days during a year |
| `neighbourhood_group` | Categorical | NYC borough/group |
| `neighbourhood` | Categorical | Specific NYC neighbourhood |

### Prediction classes

The model predicts one of:

```text
Entire home/apt
Private room
Shared room
```

---

## 🧹 Data Cleaning

The notebook performs several data preparation steps before model training.

### Removed unnecessary columns

The following columns were removed because they were not used as prediction features:

```python
['id', 'name', 'host_id', 'host_name', 'last_review']
```

### Missing values

`reviews_per_month` is filled with `0` because a listing with no reviews can reasonably have zero reviews per month.

```python
df_clean['reviews_per_month'] = df_clean['reviews_per_month'].fillna(0)
```

### Outlier handling

Extreme values for `price` and `minimum_nights` are capped at their respective 99th percentiles instead of removing complete rows.

---

## ⚙️ Preprocessing Pipeline

The project uses scikit-learn's `ColumnTransformer` and `Pipeline` so that preprocessing and prediction use the same transformations.

### Numerical features

The numerical pipeline contains:

1. Median imputation
2. Yeo-Johnson power transformation
3. Standard scaling

```python
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('skewness', PowerTransformer(method='yeo-johnson')),
    ('scaler', StandardScaler())
])
```

### Categorical features

The categorical pipeline contains:

1. Most-frequent imputation
2. One-hot encoding
3. Unknown-category handling

```python
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])
```

This approach allows the complete preprocessing and model architecture to be saved together as one reusable pipeline.

---

## 🤖 Model Development

Several classification algorithms were compared during development:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Model comparison considered both:

- Accuracy
- Macro F1 score

Macro F1 was included because the target classes are imbalanced and macro averaging gives each class equal importance.

### Final model

The final model uses a **Random Forest Classifier** with class balancing:

```python
RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)
```

### Hyperparameter tuning

`RandomizedSearchCV` was used to search over Random Forest hyperparameters including:

- `n_estimators`
- `max_depth`
- `min_samples_split`

The search used 3-fold cross-validation and optimized **macro F1**.

---

## 💾 Saved Model

The final trained pipeline is saved as:

```text
Model_Pipeline.pkl
```

It contains the preprocessing steps and trained Random Forest model required to transform new input data and produce predictions.

The API loads it with:

```python
model = joblib.load("Model_Pipeline.pkl")
```

Keeping preprocessing and the classifier together helps ensure that prediction-time data goes through the same transformations used during training.

---

# 🌐 FastAPI Backend

The backend is implemented in:

```text
main.py
```

The API provides two main endpoints.

## GET `/`

Used as a simple health check.

Example response:

```text
Hello, welcome to the NYC Room Type Prediction API!
```

## POST `/predict`

Receives listing information and returns the predicted room type and class probabilities.

### Example request

```json
{
  "latitude": 40.7484,
  "longitude": -73.9857,
  "price": 120,
  "minimum_nights": 2,
  "number_of_reviews": 84,
  "reviews_per_month": 2.3,
  "calculated_host_listings_count": 1,
  "availability_365": 210,
  "neighbourhood_group": "Manhattan",
  "neighbourhood": "Midtown"
}
```

### Example response

```json
{
  "predicted_room_type": "Entire home/apt",
  "probability": [
    0.85,
    0.14,
    0.01
  ]
}
```

The exact probability values depend on the trained model and input data.

---

# 🖥️ Frontend

The frontend consists of:

```text
index.html
style.css
script.js
```

### `index.html`

Provides the prediction form and result interface.

### `style.css`

Contains the visual design, layout, animations, building visualization, probability bars, and responsive styling.

### `script.js`

Handles:

- Form validation
- Input collection
- JSON payload generation
- API requests
- Prediction response processing
- Probability visualization
- API health checking
- Example inputs

The frontend sends requests to:

```javascript
const API_BASE_URL = "https://nyc-house-type-project-ml.onrender.com";
```

and sends prediction requests to:

```text
POST /predict
```

---

# 🔗 Frontend → API Communication

The application works through the following request flow:

```text
User enters listing information
          ↓
JavaScript collects form values
          ↓
Values converted into JSON
          ↓
POST request to FastAPI /predict
          ↓
Pydantic validates the input
          ↓
Pandas creates a one-row DataFrame
          ↓
Saved ML pipeline processes the data
          ↓
Random Forest predicts room type
          ↓
API returns prediction + probabilities
          ↓
JavaScript renders the result
```

---

# 🚀 Local Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd NYC_House_Type_project_ML
```

## 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Start the FastAPI server

```bash
uvicorn main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

FastAPI's interactive documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

---

# ☁️ Render Deployment

The backend can be deployed to Render as a Python web service.

### Recommended build command

```bash
pip install -r requirements.txt
```

### Recommended start command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

The application uses `$PORT` so that Render can assign the correct service port.

After deployment, the API base URL should be configured in `script.js`:

```javascript
const API_BASE_URL = "https://YOUR-RENDER-SERVICE.onrender.com";
```

Then the frontend sends prediction requests to:

```text
https://YOUR-RENDER-SERVICE.onrender.com/predict
```

---

# 🛠️ Important Deployment Notes

### 1. Keep the model file in the deployed project

The backend loads:

```text
Model_Pipeline.pkl
```

Therefore the file must exist in the application's working directory when the server starts.

### 2. Keep frontend and backend API URLs synchronized

If the Render service URL changes, update:

```javascript
const API_BASE_URL = "...";
```

in `script.js`.

### 3. CORS

The FastAPI application currently enables CORS so that the frontend can communicate with the API from another origin.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For a production application, CORS can be restricted to the specific frontend domain.

### 4. API response field names

The frontend must use the exact field names returned by FastAPI because JavaScript property names are case-sensitive.

The API returns:

```json
{
  "predicted_room_type": "...",
  "probability": []
}
```

The frontend should therefore read:

```javascript
result.predicted_room_type
result.probability
```

not differently capitalized property names.

---

# 📁 Project Structure

```text
NYC_House_Type_project_ML/
│
├── AB_NYC_2019.csv             # NYC Airbnb dataset
├── Model_Pipeline.pkl          # Trained ML pipeline
├── nyc_room_type.ipynb         # Data analysis and model development notebook
│
├── main.py                     # FastAPI backend
├── requirements.txt            # Python dependencies
│
├── index.html                  # Frontend HTML
├── style.css                   # Frontend styling
├── script.js                   # Frontend/API logic
│
├── .gitignore                  # Git ignored files
├── .python-version             # Python version configuration
├── .gitattributes              # Git attributes
└── README.md                   # Project documentation
```

---

# 🧪 Testing the API with Postman

You can test the API independently of the frontend.

### Method

```text
POST
```

### URL

```text
https://YOUR-RENDER-SERVICE.onrender.com/predict
```

### Header

```text
Content-Type: application/json
```

### Body → raw → JSON

```json
{
  "latitude": 40.7484,
  "longitude": -73.9857,
  "price": 120,
  "minimum_nights": 2,
  "number_of_reviews": 84,
  "reviews_per_month": 2.3,
  "calculated_host_listings_count": 1,
  "availability_365": 210,
  "neighbourhood_group": "Manhattan",
  "neighbourhood": "Midtown"
}
```

If the API returns a prediction in Postman but the website does not display it, the problem is likely in the frontend JavaScript rather than the machine learning model or FastAPI backend.

---

# 🔍 API Validation

FastAPI uses Pydantic validation for the prediction request.

Examples of validation rules include:

- Latitude must be between `-90` and `90`
- Longitude must be between `-180` and `180`
- Price must be greater than `0`
- Minimum nights must be between `1` and `365`
- Reviews cannot be negative
- Reviews per month cannot be negative
- Availability must be between `0` and `365`
- Neighbourhood and neighbourhood group cannot be empty

Invalid input is returned by FastAPI as a validation error instead of being sent to the model.

---

# 📈 Evaluation Strategy

Because the target variable is not perfectly balanced, the project does not rely exclusively on accuracy.

The model-development process considers:

### Accuracy

Measures the overall percentage of correctly classified listings.

### Macro F1

Calculates F1 separately for each class and then averages the class scores. This prevents the largest class from completely dominating the evaluation metric.

### Confusion Matrix

The notebook also generates a confusion matrix to examine how predictions are distributed across the three room-type classes.

---

# 🔮 Future Improvements

Possible future improvements include:

- Add more advanced model comparison
- Evaluate additional metrics such as weighted F1 and balanced accuracy
- Add SHAP or other explainability techniques
- Display feature-level explanations for individual predictions
- Add model versioning
- Restrict production CORS to the frontend domain
- Add automated tests for the API
- Add CI/CD using GitHub Actions
- Add structured API logging and monitoring
- Containerize the application with Docker
- Add a dedicated frontend hosting platform
- Add a map-based location input
- Improve accessibility and mobile responsiveness

---

# 🧰 Technologies Used

### Programming

- Python
- JavaScript
- HTML
- CSS

### Machine Learning

- Pandas
- NumPy
- Scikit-learn
- Joblib

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Data Analysis

- Matplotlib
- Seaborn
- Jupyter Notebook

### Deployment

- Render
- Git / GitHub

---

# 👨‍💻 Author

**Ruhi Tahmidul Islam Al Rashedi**

Data Science and Machine Learning professional with interests in machine learning, data analysis, explainable AI, and practical ML applications.

---

# 📄 License

This project is intended for educational, portfolio, and machine learning demonstration purposes. If you reuse the dataset, follow the dataset provider's applicable terms and attribution requirements.

---

## ⭐ Project Summary

This project demonstrates an end-to-end machine learning workflow:

**Data → Cleaning → EDA → Feature Engineering → Preprocessing → Model Selection → Hyperparameter Tuning → Evaluation → Model Serialization → FastAPI → Web Frontend → Cloud Deployment**

It provides a practical example of how a trained machine learning model can be converted into an interactive web application and exposed through a REST API.
