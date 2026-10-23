# Rentwise — House Rent Predictor

Rentwise is a full-stack machine-learning web application that estimates monthly house rent in Indian rupees from property details such as city, locality, BHK, size, bathrooms, floor, area type, furnishing status, tenant preference, and point of contact.

**Live application:** https://rentwise-frontend-rlvy.onrender.com  
**Backend API docs:** https://rentwise-api-hqye.onrender.com/docs

> **Note:** This is a machine-learning estimate, not a guaranteed market price or a professional property valuation. Results depend on the dataset and the information entered.

## Table of contents

- [Features](#features)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Use the live app](#use-the-live-app)
- [Run locally](#run-locally)
- [API reference](#api-reference)
- [Deployment configuration](#deployment-configuration)
- [Dataset and model notes](#dataset-and-model-notes)
- [Troubleshooting](#troubleshooting)

## Features

- Responsive Next.js interface for entering property details.
- FastAPI backend that accepts property information and returns an estimated monthly rent.
- Gradient Boosting regression model based on the project notebook's final modelling pipeline.
- Feature preparation for floor information, posting month, and locality frequency.
- Input validation for required property fields and floor ranges.
- Public frontend and API deployments hosted on Render.

## How it works

1. Enter the property's location and details in the web form.
2. The frontend sends a JSON request to the backend's `POST /predict` endpoint.
3. The backend prepares the model features, fits the model from the repository dataset when needed, and generates a prediction.
4. The API returns the estimated monthly rent as a numeric `predicted_rent` value in INR.
5. The frontend displays the estimate.

The API caches the fitted model in process memory, so the first prediction after a cold start can take longer than later requests. Render's free service may also spin down after inactivity, which can make the first request slower.

## Tech stack

**Frontend**
- Next.js 15
- React 19
- TypeScript
- Tailwind CSS 4

**Backend and machine learning**
- Python 3.13.3 (pinned in `.python-version`)
- FastAPI and Uvicorn
- NumPy and pandas
- scikit-learn
- Gradient Boosting regression

**Hosting**
- Render for the frontend and backend

## Project structure

```text
Rent_prediction_model/
├── app/                     # Next.js App Router pages and layout
├── backend/
│   └── main.py              # FastAPI application and prediction pipeline
├── House_prediction.csv     # Dataset used by the API
├── rental-price-predictor.ipynb
├── requirements.txt         # Python dependencies
├── package.json             # Frontend scripts and dependencies
├── .python-version          # Python runtime pin for deployment
└── README.md
```

## Use the live app

Open **https://rentwise-frontend-rlvy.onrender.com**, enter the property details, and select the prediction button to view an estimated monthly rent.

## Run locally

You need Node.js/npm and Python 3.13.3 installed.

### 1. Clone the repository

```bash
git clone https://github.com/SachinSingh-01/Rent_prediction_model.git
cd Rent_prediction_model
```

### 2. Configure and run the backend

Create a virtual environment from the repository root.

**Windows PowerShell**
```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

**macOS/Linux**
```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The API documentation is available at http://127.0.0.1:8000/docs.

### 3. Configure and run the frontend

Open a second terminal in the repository root and create a `.env.local` file containing:

```env
NEXT_PUBLIC_PREDICTION_API_URL=http://127.0.0.1:8000
```

Install the frontend dependencies and start Next.js:

```bash
npm install
npm run dev
```

Open http://localhost:3000.

**Keep `.env.local` local. Do not commit it to GitHub.** Next.js reads `NEXT_PUBLIC_*` variables at build time, so restart the development server after changing the variable. For production, configure the variable in the hosting provider's environment settings before building the frontend.

### 4. Useful frontend commands

```bash
npm run dev
npm run lint
npm run build
npm run start
```

## API reference

- Interactive API documentation: https://rentwise-api-hqye.onrender.com/docs
- OpenAPI schema: https://rentwise-api-hqye.onrender.com/openapi.json
- Prediction endpoint: `POST https://rentwise-api-hqye.onrender.com/predict`

Example request:

```json
{
  "city": "Mumbai",
  "locality": "Andheri",
  "bhk": 2,
  "size": 850,
  "bathrooms": 2,
  "floor": 3,
  "total_floors": 10,
  "area_type": "Super Area",
  "furnishing_status": "Semi-Furnished",
  "tenant_preferred": "Bachelors/Family",
  "point_of_contact": "Contact Owner"
}
```

Successful responses contain a numeric estimate, for example:

```json
{
  "predicted_rent": 25000.0
}
```

The number above illustrates the response format only; it is not a prediction for the example request. Valid categories and city/locality values should reflect the dataset and the model's training data. Invalid request fields or floor values can produce a validation error.

## Deployment configuration

The current public deployment uses Render for both services.

### Frontend service

Set the following environment variable in the frontend service:

```env
NEXT_PUBLIC_PREDICTION_API_URL=https://rentwise-api-hqye.onrender.com
```

Build command:

```bash
npm install && npm run build
```

Start command:

```bash
npm run start
```

### Backend service

The backend service uses the Python version in `.python-version` and the dependencies in `requirements.txt`.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Set the backend environment variable `RENTWISE_CORS_ORIGINS` to the exact deployed frontend origin:

```env
RENTWISE_CORS_ORIGINS=https://rentwise-frontend-rlvy.onrender.com
```

If the frontend domain changes, update this value to match the new origin. Do not add a trailing slash to the origin.

## Dataset and model notes

- The API reads `House_prediction.csv` from the repository root.
- The model pipeline follows the final modelling approach in `rental-price-predictor.ipynb`, using feature preparation, median imputation, one-hot encoding, and a Gradient Boosting regressor.
- The target is transformed with a logarithm during training and converted back to the rent scale for predictions.
- The API derives the posting month from the backend process's current local month because the frontend request does not include a posting date.
- Locality frequencies are derived from the training partition; an unseen locality uses a fallback frequency.
- The fitted model is cached in process memory rather than loaded from a saved model artifact. A cold start may therefore require model fitting again.

## Troubleshooting

**The first request is slow:** The Render service may be waking from inactivity, and the API may need to fit the model.

**The frontend says “Failed to fetch”:** Confirm that `NEXT_PUBLIC_PREDICTION_API_URL` points to the deployed backend and that `RENTWISE_CORS_ORIGINS` includes the exact frontend origin. Redeploy the frontend after changing its `NEXT_PUBLIC_*` variable.

**The API is unavailable:** Open the API docs link above and check the Render service logs and deployment status.

---

Built as a learning project to practise Python, data preparation, regression modelling, API development, and full-stack deployment.
