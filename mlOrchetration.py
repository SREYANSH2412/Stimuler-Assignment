from ray import serve
import ray
from transformers import (
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    AutoTokenizer, AutoModelForCausalLM
)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import torch
import uvicorn
import time

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@serve.deployment
@serve.ingress(app)
class ModelPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize BERT summarizer
        print("Loading BERT model...")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained(
            "bert-base-uncased",
            output_attentions=True  # Enable attention outputs
        ).to(self.device)
        
        # Initialize RoBERTa sentiment analyzer
        print("Loading RoBERTa model...")
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.roberta_model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=2  # Binary classification
        ).to(self.device)
        
        # Initialize DistilBERT sentiment analyzer
        print("Loading DistilBERT model...")
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.distilbert_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2  # Binary classification
        ).to(self.device)
        
        # Initialize GPT-2
        print("Loading GPT-2 model...")
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
        
        print("All models loaded successfully!")

    async def summarize(self, text: str) -> Dict[str, str]:
        # Ensure text has at least one sentence
        sentences = text.split('.')
        if len(sentences) < 2:
            return {"summary": text}
            
        inputs = self.bert_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs, output_attentions=True)
            
        # Get the last layer's attention
        last_attention = outputs.attentions[-1]
        # Average attention across heads
        averaged_attention = last_attention.mean(dim=1)
        # Get sentence-level attention scores
        sentence_scores = averaged_attention.mean(dim=-1).squeeze()
        
        # Select the sentence with highest attention score
        if len(sentence_scores) > 0:
            best_sentence_idx = min(int(sentence_scores.argmax()), len(sentences) - 1)
            summary = sentences[best_sentence_idx].strip()
            if not summary:  # If empty summary, return first sentence
                summary = sentences[0].strip()
        else:
            summary = sentences[0].strip()
            
        return {"summary": summary}

    async def analyze_sentiment(self, text: str, model_type: str) -> float:
        if model_type == "roberta":
            tokenizer = self.roberta_tokenizer
            model = self.roberta_model
        else:
            tokenizer = self.distilbert_tokenizer
            model = self.distilbert_model
            
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        scores = torch.softmax(outputs.logits, dim=1)
        return scores[0][1].item()

    async def generate_response(self, text: str) -> Dict[str, str]:
        inputs = self.gpt2_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.gpt2_model.generate(
                **inputs,
                max_length=150,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7
            )
        
        response = self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_response": response}

    @app.post("/process")
    async def process_text(self, request: TextRequest) -> Dict[str, Any]:
        try:
            print(f"Processing request with text: {request.text[:100]}...")  # Print first 100 chars
            
            # Get summary
            summary_result = await self.summarize(request.text)
            summary = summary_result["summary"]
            print(f"Generated summary: {summary}")
            
            # Get sentiment from both models
            roberta_sentiment = await self.analyze_sentiment(request.text, "roberta")
            distilbert_sentiment = await self.analyze_sentiment(request.text, "distilbert")
            avg_sentiment = (roberta_sentiment + distilbert_sentiment) / 2
            print(f"Sentiment scores - RoBERTa: {roberta_sentiment:.2f}, DistilBERT: {distilbert_sentiment:.2f}")
            
            # Generate response
            llm_response = await self.generate_response(summary)
            print(f"Generated response: {llm_response['generated_response'][:100]}...")
            
            result = {
                "summary": summary,
                "sentiment": {
                    "roberta": roberta_sentiment,
                    "distilbert": distilbert_sentiment,
                    "average": avg_sentiment
                },
                "generated_response": llm_response["generated_response"]
            }
            
            print("Request processed successfully!")
            return result
            
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

def main():
    # Initialize Ray if it hasn't been initialized
    if not ray.is_initialized():
        ray.init()
    
    # Start Ray Serve
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    
    # Deploy the application
    deployment = serve.run(ModelPipeline.bind())
    
    print("Service is running at http://localhost:8000")
    print("To test the service, send a POST request to http://localhost:8000/process")
    print("Example curl command:")
    print('''curl -X POST "http://localhost:8000/process" -H "Content-Type: application/json" -d "{\\"text\\":\\"Your text here for processing.\\"}"''')
    
    # Keep the program running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down the service...")
        serve.shutdown()

if __name__ == "__main__":
    main()