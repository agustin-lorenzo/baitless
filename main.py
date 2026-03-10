from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

binary_pipe = pipeline('text-classification', model='models/f-binary_db-model')
class_pipe = pipeline('text-classification', model='models/f-class_model', top_k=3)

app = FastAPI()

# Validate input data
class Inputs(BaseModel):
    text: str
    
@app.post('/predict')
def predict_fallacy(inputs: Inputs):
    binary_pred = binary_pipe(inputs.text)
    class_pred = class_pipe(inputs.text)
    top3 = sorted(class_pred[0], key=lambda x: x['score'], reverse=True)[:3]
    return {'detected': binary_pred[0]['label'],
            'top_predictions': top3}

@app.get('/')
def read_root():
    return {'message': 'welcome to the fallacy api'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)


# from flask import Flask, request, jsonify

# app = Flask(__name__)

# @app.route("/get-user/<user_id>")
# def get_user(user_id):
#     user_data = {
#         "user_id": user_id
#     }

#     extra = request.args.get("extra")
#     if extra:
#         user_data["extra"] = extra

#     return jsonify(user_data), 200

# @app.route("/create-user", methods=["POST"])
# def create_user():
#     data = request.get_json()
#     return jsonify(data), 201


# if __name__ == "__main__":
#     app.run(debug=True)
