from transformers import pipeline
import subprocess

global emotion_model
emotion_model = pipeline('sentiment-analysis', 
                        model='arpanghoshal/EmoRoBERTa')


#will automatically download required corupus
subprocess.run(["python", "-m", "textblob.download_corpora"])

