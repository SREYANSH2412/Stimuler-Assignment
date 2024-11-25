import requests

# Test text
text = """
I recently purchased the new smartphone and I'm extremely impressed with its performance. 
The camera quality is outstanding and the battery life is much better than my previous phone. 
However, the price point is quite high. Overall, I would recommend this phone to others.
"""

# Make request to the API
response = requests.post(
    "http://localhost:8001/process",
    params={"text": text}  # Changed to use json parameter instead of params
)

# Print results
if response.status_code == 200:
    result = response.json()
    print("\nSummary:")
    print(result['summary'])
    print("\nSentiment Analysis:")
    print(f"RoBERTa Score: {result['sentiment']['roberta']:.2f}")
    print(f"DistilBERT Score: {result['sentiment']['distilbert']:.2f}")
    print(f"Average Sentiment: {result['sentiment']['average']:.2f}")
    print("\nGenerated Response:")
    print(result['generated_response'])
else:
    print(f"Error: {response.status_code}")
    print(response.text)