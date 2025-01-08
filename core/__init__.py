from transformers import pipeline
import subprocess
import os

HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")

if not HF_AUTH_TOKEN:
    print("HF_AUTH_TOKEN is not found in the environment!")
else:
    # Just show partial token to confirm we got something:
    print(f"HF_AUTH_TOKEN found, starts with: {HF_AUTH_TOKEN[:10]}...")
global emotion_model
emotion_model = pipeline('sentiment-analysis', 
                        model='arpanghoshal/EmoRoBERTa')


#will automatically download required corupus
subprocess.run(["python", "-m", "textblob.download_corpora"])

