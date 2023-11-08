import base64

from fastapi import FastAPI, UploadFile
from backend.api.estimate import main

app = FastAPI()

@app.post("/estimate")
async def root(file: UploadFile):
    contents = await file.read()
    predicate = main.estimate(contents)
    return {'predict': predicate}
