import re

def clean_text(text):
    text = str(text).lower()
    
    # remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # remove punctuation
    import string
    text = "".join([c for c in text if c not in string.punctuation])
    
    # remove extra spaces
    text = " ".join(text.split())
    
    return text