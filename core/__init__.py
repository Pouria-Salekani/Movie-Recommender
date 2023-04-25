from transformers import pipeline
import subprocess

global emotion_model
emotion_model = pipeline('sentiment-analysis', 
                        model='arpanghoshal/EmoRoBERTa')


subprocess.run(["python", "-m", "textblob.download_corpora"])

