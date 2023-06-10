#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: christos
"""


# from fastapi import FastAPI
# import os
# import sys

# sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

# # No need to rename `app` to `application`
# from fast_api_app_creation_04 import app as application_v1

# app = FastAPI()


# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Main API"}


# # Mount the application from fast_api_app_creation_04.py under the "/v1" route
# app.mount("/v1", application_v1)

# main.py

from fastapi import FastAPI
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from fast_api_app_creation_04 import router as application_v1_router

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to the Main API"}


app.include_router(application_v1_router, prefix="/v1")
