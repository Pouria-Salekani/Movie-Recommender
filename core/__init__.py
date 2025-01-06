from transformers import pipeline
import subprocess

global emotion_model
emotion_model = pipeline('sentiment-analysis', 
                        model='arpanghoshal/EmoRoBERTa', use_auth_token=True)


#will automatically download required corupus
subprocess.run(["python", "-m", "textblob.download_corpora"])

