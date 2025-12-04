# service/app/auth.py
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
import os

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def require_api_key(api_key: str = Security(api_key_header)):
    expected = os.getenv("INFERENCE_API_KEY")
    if expected is None:
        raise HTTPException(status_code=500, detail="INFERENCE_API_KEY not configured")
    if not api_key or api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True
