from models import Argument, ArgumentResponse
from fastapi import FastAPI
import pickle
import numpy as np 


app = FastAPI(
    title="Titanic API",
    version="0.1",
    description="Kader belirleyn servis"
)

global model
f = open('ml_model/rf.pkl', 'rb')
model = pickle.load(f)
f.close()


async def predict(sex, age, sib_count, pclass):
    result = model.predict(np.array([sex, age, sib_count, pclass]).reshape(1,-1))[0]
    res_proba = model.predict_proba(np.array([sex, age, sib_count, pclass]).reshape(1,-1))[0]
    if result == 0:
        return (0, res_proba[0]*100)
    else:
        return (1, res_proba[1]*100)


@app.post('/predict', summary="Titanik kazasına karışsaydın hayatta kalır mıydın?")
async def predict_survive(arg: Argument):
    results = await predict(arg.sex, arg.age, arg.sib_count, arg.pclass)
    response = ArgumentResponse(survive=results[0], proba=results[1])
    return response

