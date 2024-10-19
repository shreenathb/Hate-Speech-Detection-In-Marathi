from fastapi import FastAPI
import uvicorn
import pickle
from pydantic import BaseModel

class TextData(BaseModel):
    text : str

app = FastAPI()

pickle_in = open("model.pkl", "rb")
classifier = pickle.load(pickle_in)

vec_in = open('vectorizer.pkl', 'rb')
vectorizer = pickle.load(vec_in)

@app.post("/predict")
def predict(data : TextData):
    
    text = data.text

    vectorized_text = vectorizer.transform([text])
    prediction = classifier.predict(vectorized_text)

    return {"text": text, "prediction": int(prediction[0])}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port = 8000)