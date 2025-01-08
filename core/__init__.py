from transformers import pipeline
import subprocess

global emotion_model
emotion_model = pipeline('sentiment-analysis', 
                        model='arpanghoshal/EmoRoBERTa', use_auth_token='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')



# from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
# import subprocess
# import os

# # HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")
# # print(HF_AUTH_TOKEN)

# global emotion_model
# tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
# model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

# emotion_model = pipeline('sentiment-analysis', 
#                     model='arpanghoshal/EmoRoBERTa', use_auth_token='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'
# )

#will automatically download required corupus
subprocess.run(["python", "-m", "textblob.download_corpora"])

