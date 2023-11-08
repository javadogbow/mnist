import base64

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from backend.api.estimate import main

app = FastAPI()

origins = [
    'http://localhost:5174'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_methods = ['*'],
    allow_headers = ['*']
)

@app.post("/estimate")
async def root(file: UploadFile):
    contents = await file.read()
    predicate = main.estimate(contents)
    return {'predict': predicate}
