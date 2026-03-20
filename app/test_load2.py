import sys
import os
sys.path.append(r"d:\Final_Project1 - Copy\app")
from app import load_resources, hybrid_model

print("Testing model load...")
load_resources()

if hybrid_model is not None:
    print("SUCCESS: Model loaded!")
else:
    print("FAILURE: Model is still None!")
