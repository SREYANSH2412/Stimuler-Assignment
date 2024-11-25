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
from typing import Dict, Any, List
import torch
import torch.nn.parallel
import uvicorn
import time
import numpy as np
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@dataclass
class GPUStats:
    device_id: int
    memory_used: float
    memory_total: float
    utilization: float

class GPUManager:
    def __init__(self):
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus == 0:
            logger.warning("No GPUs available. Running on CPU.")
            self.device = torch.device("cpu")
        else:
            logger.info(f"Found {self.num_gpus} GPUs")
            self.device = torch.device("cuda")
            
    def get_gpu_stats(self) -> List[GPUStats]:
        if self.num_gpus == 0:
            return []
            
        stats = []
        for i in range(self.num_gpus):
            memory_used = torch.cuda.memory_allocated(i) / (1024**3)  # Convert to GB
            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            utilization = memory_used / memory_total * 100
            stats.append(GPUStats(i, memory_used, memory_total, utilization))
        return stats
        
    def get_least_utilized_gpu(self) -> int:
        if self.num_gpus == 0:
            return -1
            
        stats = self.get_gpu_stats()
        return min(stats, key=lambda x: x.utilization).device_id
        
    def get_optimal_device(self, model_size_gb: float) -> torch.device:
        if self.num_gpus == 0:
            return torch.device("cpu")
            
        gpu_id = self.get_least_utilized_gpu()
        stats = self.get_gpu_stats()[gpu_id]
        
        # If GPU has enough free memory, use it
        if stats.memory_total - stats.memory_used > model_size_gb:
            return torch.device(f"cuda:{gpu_id}")
        else:
            logger.warning(f"Insufficient GPU memory. Using CPU for computation.")
            return torch.device("cpu")

@serve.deployment(ray_actor_options={"num_gpus": 1})  # Request GPU resources from Ray
@serve.ingress(app)
class ModelPipeline:
    def __init__(self):
        self.gpu_manager = GPUManager()
        logger.info(f"Initializing ModelPipeline with {self.gpu_manager.num_gpus} GPUs")
        
        # Estimate model sizes (approximate values in GB)
        self.model_sizes = {
            "bert": 0.5,
            "roberta": 0.5,
            "distilbert": 0.25,
            "gpt2": 0.5
        }
        
        # Initialize models with optimal GPU placement
        self.init_bert()
        self.init_sentiment_analyzers()
        self.init_gpt2()
        
        logger.info("All models loaded successfully!")
        self.log_gpu_status()

    def init_bert(self):
        logger.info("Loading BERT model...")
        device = self.gpu_manager.get_optimal_device(self.model_sizes["bert"])
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained(
            "bert-base-uncased",
            output_attentions=True
        ).to(device)
        self.bert_device = device
        
        # Enable model parallelism if multiple GPUs available
        if self.gpu_manager.num_gpus > 1:
            self.bert_model = torch.nn.DataParallel(self.bert_model)
        
    def init_sentiment_analyzers(self):
        logger.info("Loading sentiment analysis models...")
        
        # RoBERTa initialization
        roberta_device = self.gpu_manager.get_optimal_device(self.model_sizes["roberta"])
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.roberta_model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=2
        ).to(roberta_device)
        self.roberta_device = roberta_device
        
        # DistilBERT initialization
        distilbert_device = self.gpu_manager.get_optimal_device(self.model_sizes["distilbert"])
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.distilbert_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2
        ).to(distilbert_device)
        self.distilbert_device = distilbert_device
        
        # Enable model parallelism if multiple GPUs available
        if self.gpu_manager.num_gpus > 1:
            self.roberta_model = torch.nn.DataParallel(self.roberta_model)
            self.distilbert_model = torch.nn.DataParallel(self.distilbert_model)
            
    def init_gpt2(self):
        logger.info("Loading GPT-2 model...")
        device = self.gpu_manager.get_optimal_device(self.model_sizes["gpt2"])
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        self.gpt2_device = device
        
        # Enable model parallelism if multiple GPUs available
        if self.gpu_manager.num_gpus > 1:
            self.gpt2_model = torch.nn.DataParallel(self.gpt2_model)

    def log_gpu_status(self):
        stats = self.gpu_manager.get_gpu_stats()
        for stat in stats:
            logger.info(f"GPU {stat.device_id}: {stat.memory_used:.2f}GB/{stat.memory_total:.2f}GB "
                       f"({stat.utilization:.1f}% utilized)")

    async def summarize(self, text: str) -> Dict[str, str]:
        sentences = text.split('.')
        if len(sentences) < 2:
            return {"summary": text}
            
        inputs = self.bert_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.bert_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs, output_attentions=True)
            
        last_attention = outputs.attentions[-1]
        averaged_attention = last_attention.mean(dim=1)
        sentence_scores = averaged_attention.mean(dim=-1).squeeze()
        
        if len(sentence_scores) > 0:
            best_sentence_idx = min(int(sentence_scores.argmax()), len(sentences) - 1)
            summary = sentences[best_sentence_idx].strip()
            if not summary:
                summary = sentences[0].strip()
        else:
            summary = sentences[0].strip()
            
        return {"summary": summary}

    async def analyze_sentiment(self, text: str, model_type: str) -> float:
        if model_type == "roberta":
            tokenizer = self.roberta_tokenizer
            model = self.roberta_model
            device = self.roberta_device
        else:
            tokenizer = self.distilbert_tokenizer
            model = self.distilbert_model
            device = self.distilbert_device
            
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        scores = torch.softmax(outputs.logits, dim=1)
        return scores[0][1].item()

    async def generate_response(self, text: str) -> Dict[str, str]:
        inputs = self.gpt2_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.gpt2_device) for k, v in inputs.items()}
        
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
            start_time = time.time()
            logger.info(f"Processing request with text: {request.text[:100]}...")
            
            # Log GPU status before processing
            self.log_gpu_status()
            
            # Get summary
            summary_result = await self.summarize(request.text)
            summary = summary_result["summary"]
            logger.info(f"Generated summary: {summary}")
            
            # Run sentiment analysis in parallel using different GPUs
            roberta_sentiment = await self.analyze_sentiment(request.text, "roberta")
            distilbert_sentiment = await self.analyze_sentiment(request.text, "distilbert")
            avg_sentiment = (roberta_sentiment + distilbert_sentiment) / 2
            logger.info(f"Sentiment scores - RoBERTa: {roberta_sentiment:.2f}, "
                       f"DistilBERT: {distilbert_sentiment:.2f}")
            
            # Generate response
            llm_response = await self.generate_response(summary)
            logger.info(f"Generated response: {llm_response['generated_response'][:100]}...")
            
            result = {
                "summary": summary,
                "sentiment": {
                    "roberta": roberta_sentiment,
                    "distilbert": distilbert_sentiment,
                    "average": avg_sentiment
                },
                "generated_response": llm_response["generated_response"],
                "processing_time": time.time() - start_time
            }
            
            # Log final GPU status
            self.log_gpu_status()
            logger.info("Request processed successfully!")
            return result
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

def main():
    if not ray.is_initialized():
        ray.init()
    
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    
    deployment = serve.run(ModelPipeline.bind())
    
    logger.info("Service is running at http://localhost:8000")
    logger.info("To test the service, send a POST request to http://localhost:8000/process")
    logger.info('''Example curl command:
    curl -X POST "http://localhost:8000/process" -H "Content-Type: application/json" -d "{\\"text\\":\\"Your text here for processing.\\"}"''')
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down the service...")
        serve.shutdown()

if __name__ == "__main__":
    main()