NFL ML Prediction Server

This is the Python ML Prediction Server required by Lovable's Phase 4 ML integration. It implements the exact API contract expected by the ml-client and ml-predict Edge Function.

This service exposes the following endpoints:

GET /v1/health

POST /v1/predict/pregame

POST /v1/predict/ensemble

POST /v1/predict/vegas-edge

POST /v1/predict/batch

These endpoints return responses in the required MLPredictResponse format used by Lovable AI.

Run Locally

Install dependencies:
pip install -r requirements.txt

Start the server:
uvicorn server:app --reload

Your local URLs will be:
http://127.0.0.1:8000/v1/health

http://127.0.0.1:8000/v1/predict/pregame

Deploy to Render

Create a new Web Service on Render.

Connect this GitHub repo.

Set Build Command:
pip install -r requirements.txt

Set Start Command:
uvicorn server:app --host 0.0.0.0 --port $PORT

Deploy the service.

Your deployed ML_SERVICE_URL will look like:
https://nfl-ml-server.onrender.com

ML Service URL for Lovable

After deployment, set:
ML_SERVICE_URL = https://your-render-url

(Optional) If you enable API key protection, set:
ML_API_KEY = <your-secret>

Notes

This server currently uses placeholder ML math for predictions.

It fully satisfies the API contract required by Lovable for Phase 4 integration.

Real XGBoost / LightGBM can be added later inside base_prediction() and ensemble helpers.
