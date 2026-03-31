from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import pipeline
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

binary_pipe = pipeline('text-classification', model='agustin-lorenzo/fallacy-detector_db')
class_pipe = pipeline('text-classification', model='agustin-lorenzo/fallacy-classifier', top_k=3)


# Validate input data
class Inputs(BaseModel):
    text: str
    
@app.post('/predict')
@limiter.limit('10/minute')
def predict_fallacy(request: Request, inputs: Inputs):
    start_time = time.time()
    logger.info(f"Request from {request.client.host} | text length: {len(inputs.text)}")
    
    if not inputs.text.strip():
        logger.warning(f"Empty text from {request.client.host}")
        raise HTTPException(status_code=422, detail="Text cannot be empty.")
    if len(inputs.text.strip()) > 5000:
        logger.warning(f"Text too long from {request.client.host} | Text length: {len(inputs.text)}")
        raise HTTPException(status_code=422, detail="Text cannot be longer than 500 characters.")
    
    try:
        binary_pred = binary_pipe(inputs.text)
        class_pred = class_pipe(inputs.text)
        top3 = sorted(class_pred[0], key=lambda x: x['score'], reverse=True)[:3]
        duration = time.time() - start_time
        logger.info(f"Prediction complete | duration: {duration:.2f}s | detected: {binary_pred[0]['label']}")
        return {'detected': binary_pred[0]['label'],
                'top_predictions': top3}
    except Exception as e:
        logger.error(f"Prediction error from {request.client.host} | error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.get('/')
def read_root():
    return {'message': 'welcome to the fallacy api'}

if __name__ == '__main__':
    import uvicorn
    import os
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
