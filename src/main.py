#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: christos
"""

from fastapi import FastAPI
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from fast_api_app_creation_04 import app as application

app = FastAPI()


@app.get("/")
def read_root():
    return {"Welcome to the Main API"}


app.mount("/v1", application)
