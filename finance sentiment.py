import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Load FinBERT
MODEL_NAME = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# FinBERT sentiment function
def get_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return pd.Series(["Neutral", 0.0])
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    probs = softmax(outputs.logits, dim=1)
    labels = ["Negative", "Neutral", "Positive"]
    sentiment_index = torch.argmax(probs).item()
    sentiment = labels[sentiment_index]
    confidence_score = probs[0][sentiment_index].item()
    return pd.Series([sentiment, confidence_score])

# File paths (✅ UPDATE these if you're using Google Colab!)
input_path = "/Users/ngocnguyen/Downloads/filtered_stocks.csv"  # already finance-only
output_path = "/Users/ngocnguyen/Desktop/CPP/Capstone/finance_sentiment_2018.csv"

# Process in chunks
chunk_iter = pd.read_csv(input_path, chunksize=10000)
count = 0

for chunk in chunk_iter:
    print(f"Processing chunk {count + 1}: {chunk.shape}")

    # Apply FinBERT sentiment to each article_title
    chunk[["Sentiment", "Confidence_Score"]] = chunk["article_title"].apply(get_sentiment)

    # Save results
    mode = "w" if count == 0 else "a"
    header = count == 0
    chunk.to_csv(output_path, index=False, mode=mode, header=header)

    count += 1

print("✅ Sentiment analysis completed for finance sector.")
print(f"📄 Results saved to: {output_path}")
