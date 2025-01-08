from transformers import pipeline
import subprocess

import environ

env = environ.Env()
environ.Env.read_env()  # reads .env file

HF_AUTH_TOKEN = env('HF_AUTH_TOKEN', default=None)

global emotion_model
emotion_model = pipeline('sentiment-analysis', 
                        model='arpanghoshal/EmoRoBERTa', use_auth_token=HF_AUTH_TOKEN)


#will automatically download required corupus
subprocess.run(["python", "-m", "textblob.download_corpora"])

