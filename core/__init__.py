from transformers import pipeline
import subprocess
import os


HF_AUTH_TOKEN = os.environ.get('HF_AUTH_TOKEN')

global emotion_model
emotion_model = pipeline('sentiment-analysis', 
                        model='arpanghoshal/EmoRoBERTa', use_auth_token=HF_AUTH_TOKEN)


#will automatically download required corupus
subprocess.run(["python", "-m", "textblob.download_corpora"])

