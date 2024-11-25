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
from typing import Dict, Any, Optional
import torch
import uvicorn
import time
import logging
import psutil
import GPUtil
from datetime import datetime
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_performance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

class TextRequest(BaseModel):
    text: str

class PerformanceMetrics:
    def __init__(self):
        self.metrics = {
            'bert_inference_times': [],
            'roberta_inference_times': [],
            'distilbert_inference_times': [],
            'gpt2_inference_times': [],
            'total_processing_times': [],
            'gpu_utilization': [],
            'memory_usage': []
        }
        
    def add_metric(self, metric_name: str, value: float):
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
            # Keep only last 1000 measurements to prevent memory bloat
            if len(self.metrics[metric_name]) > 1000:
                self.metrics[metric_name].pop(0)
            
    def get_average(self, metric_name: str) -> Optional[float]:
        if metric_name in self.metrics and self.metrics[metric_name]:
            return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
        return None
        
    def log_summary(self):
        logger.info("Performance Metrics Summary:")
        for metric_name, values in self.metrics.items():
            if values:
                avg = self.get_average(metric_name)
                logger.info(f"{metric_name}: avg={avg:.4f}, min={min(values):.4f}, max={max(values):.4f}")

@contextmanager
def timer(metrics: PerformanceMetrics, operation_name: str):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        metrics.add_metric(f"{operation_name}_inference_times", duration)
        logger.info(f"{operation_name} took {duration:.4f} seconds")

@serve.deployment
@serve.ingress(app)
class ModelPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.performance_metrics = PerformanceMetrics()
        self.initialize_models()
        
    def initialize_models(self):
        # Initialize BERT summarizer
        logger.info("Loading BERT model...")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained(
            "bert-base-uncased",
            output_attentions=True
        ).to(self.device)
        
        # Initialize RoBERTa sentiment analyzer
        logger.info("Loading RoBERTa model...")
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.roberta_model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=2
        ).to(self.device)
        
        # Initialize DistilBERT sentiment analyzer
        logger.info("Loading DistilBERT model...")
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.distilbert_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2
        ).to(self.device)
        
        # Initialize GPT-2
        logger.info("Loading GPT-2 model...")
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
        
        logger.info("All models loaded successfully!")

    def get_system_metrics(self) -> Dict[str, float]:
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'cpu_count': psutil.cpu_count(),
            'memory_available': psutil.virtual_memory().available / (1024 * 1024 * 1024)
        }
        
        if torch.cuda.is_available():
            try:
                gpu = GPUtil.getGPUs()[0]  # Get first GPU
                metrics['gpu_utilization'] = gpu.load * 100
                metrics['gpu_memory_usage'] = gpu.memoryUtil * 100
                self.performance_metrics.add_metric('gpu_utilization', metrics['gpu_utilization'])
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {str(e)}")
                
        return metrics

    async def summarize(self, text: str) -> Dict[str, str]:
        with timer(self.performance_metrics, "bert"):
            # Ensure text has at least one sentence
            sentences = text.split('.')
            if len(sentences) < 2:
                return {"summary": text}
                
            inputs = self.bert_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
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
        with timer(self.performance_metrics, model_type):
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
        with timer(self.performance_metrics, "gpt2"):
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
        start_time = time.time()
        request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        try:
            logger.info(f"Processing request {request_id} with text: {request.text[:100]}...")
            
            # Get initial system metrics
            initial_metrics = self.get_system_metrics()
            logger.info(f"Initial system metrics: {initial_metrics}")
            
            # Process with existing pipeline
            result = {
                "summary": (await self.summarize(request.text))["summary"],
                "sentiment": {
                    "roberta": await self.analyze_sentiment(request.text, "roberta"),
                    "distilbert": await self.analyze_sentiment(request.text, "distilbert")
                },
                "generated_response": (await self.generate_response(request.text))["generated_response"]
            }
            
            # Calculate average sentiment
            result["sentiment"]["average"] = (
                result["sentiment"]["roberta"] + 
                result["sentiment"]["distilbert"]
            ) / 2
            
            # Get final system metrics
            final_metrics = self.get_system_metrics()
            
            # Calculate total processing time
            total_time = time.time() - start_time
            self.performance_metrics.add_metric("total_processing_times", total_time)
            
            # Add performance data to response
            result["performance_metrics"] = {
                "processing_time": total_time,
                "system_metrics": final_metrics
            }
            
            # Log performance data
            logger.info(f"Request {request_id} processed in {total_time:.4f} seconds")
            logger.info(f"Final system metrics: {final_metrics}")
            
            # Periodically log performance summary
            if len(self.performance_metrics.metrics['total_processing_times']) % 100 == 0:
                self.performance_metrics.log_summary()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    @app.get("/metrics")
    async def get_metrics(self) -> Dict[str, Any]:
        try:
            current_metrics = self.get_system_metrics()
            avg_metrics = {
                metric_name: self.performance_metrics.get_average(metric_name)
                for metric_name in self.performance_metrics.metrics.keys()
            }
            
            return {
                "current_metrics": current_metrics,
                "average_metrics": avg_metrics
            }
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")

def main():
    # Initialize Ray if it hasn't been initialized
    if not ray.is_initialized():
        ray.init()
    # if not ray.is_initialized():
    #     try:
    #         ray.init(
    #             dashboard_host='0.0.0.0',  # Allow external access to dashboard
    #             ignore_reinit_error=True,   # Ignore if Ray is already initialized
    #             include_dashboard=True,     # Enable the dashboard
    #             _system_config={
    #                 "automatic_object_spilling_enabled": False
    #             }
    #         )
    #     except Exception as e:
    #         logger.error(f"Failed to initialize Ray: {e}")
    #         raise
    
    # Start Ray Serve
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    
    # Deploy the application
    deployment = serve.run(ModelPipeline.bind())
    
    logger.info("Service is running at http://localhost:8000")
    logger.info("To test the service, send a POST request to http://localhost:8000/process")
    logger.info("To get metrics, send a GET request to http://localhost:8000/metrics")
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