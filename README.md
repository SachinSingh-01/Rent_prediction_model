# Rentwise — House Rent Predictor

A responsive Next.js frontend for the house rent prediction project. The existing `rental-price-predictor.ipynb` notebook and `House_prediction.csv` dataset remain at the repository root and are not changed by the frontend.

## Run locally

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). To enable predictions, copy `.env.example` to `.env.local` and set `NEXT_PUBLIC_PREDICTION_API_URL` to the base URL of your deployed model API. Restart the dev server after changing environment variables.

## Model API contract

The browser sends a JSON `POST` request to `${NEXT_PUBLIC_PREDICTION_API_URL}/predict` with this body:

```json
{
  "city": "Bangalore",
  "locality": "Indiranagar",
  "bhk": 2,
  "size": 1200,
  "bathrooms": 2,
  "floor": 3,
  "total_floors": 8,
  "area_type": "Super Area",
  "furnishing_status": "Semi-Furnished",
  "tenant_preferred": "Bachelors/Family",
  "point_of_contact": "Contact Owner"
}
```

Return JSON with a numeric monthly rent in rupees, using a `predicted_rent` field (a numeric `rent` field is also accepted). The frontend does not provide a fallback prediction. Until the API URL is configured, the form displays an integration placeholder. The API must allow browser requests from the deployed Vercel domain through CORS. Add `NEXT_PUBLIC_PREDICTION_API_URL` to the Vercel project's Environment Variables to enable production predictions.

The API recreates the notebook's floor parsing (including ground and basement floor numbers), training-only locality counts, median imputation, and one-hot encoding. It fits the notebook's final log-target Gradient Boosting estimator on the same 80/20 split (`random_state=42`) and reverses the log transform with `expm1` for each estimate. The notebook and dataset stay unchanged; no model file is currently stored in the project, so the API fits the estimator from the CSV on its first prediction request.

## Checks

```bash
npm run lint
npm run build
```

## Run the prediction API locally

The FastAPI service reads the unchanged `House_prediction.csv` at the project root and fits the final Gradient Boosting model from the notebook when the first prediction is requested. The first request takes longer while training; subsequent requests reuse the in-memory fitted model.

In a second terminal, from the project root, create and activate a Python virtual environment, then install the backend dependencies:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

On macOS or Linux, use `python3 -m venv .venv` and `source .venv/bin/activate` in place of the first two commands. The API docs are available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

To connect the existing Next.js form, create `.env.local` in the project root with:

```env
NEXT_PUBLIC_PREDICTION_API_URL=http://127.0.0.1:8000
```

Restart `npm run dev` after setting the variable. The backend allows the Next.js development origins `http://localhost:3000` and `http://127.0.0.1:3000` by default. To add more allowed origins, set `RENTWISE_CORS_ORIGINS` to a comma-separated list before starting Uvicorn.

The API accepts the JSON fields emitted by the frontend and responds with `{"predicted_rent": <number>}` in INR. It derives `Posted_month` from the backend process's current local month because the frontend does not send a posting date. Locality frequencies are calculated from the notebook's training partition; unseen localities receive the notebook's fallback frequency of 1.
