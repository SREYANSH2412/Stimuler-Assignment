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
from typing import Dict, Any, Optional, List
import torch
import uvicorn
import time
import logging
import psutil
import GPUtil
from datetime import datetime
from contextlib import contextmanager
import random
from enum import Enum
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

class ExperimentVariant(Enum):
    A = "A"
    B = "B"

class TextRequest(BaseModel):
    text: str
    force_variant: Optional[ExperimentVariant] = None

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

class ExperimentMetrics:
    def __init__(self):
        self.variant_metrics = {
            ExperimentVariant.A: {
                'processing_times': [],
                'error_counts': 0,
                'total_requests': 0,
                'sentiment_scores': [],
                'summary_lengths': [],
                'response_lengths': []
            },
            ExperimentVariant.B: {
                'processing_times': [],
                'error_counts': 0,
                'total_requests': 0,
                'sentiment_scores': [],
                'summary_lengths': [],
                'response_lengths': []
            }
        }
    
    def record_metrics(self, variant: ExperimentVariant, processing_time: float, 
                      sentiment_score: float, summary_length: int,
                      response_length: int, error: bool = False):
        metrics = self.variant_metrics[variant]
        metrics['processing_times'].append(processing_time)
        metrics['sentiment_scores'].append(sentiment_score)
        metrics['summary_lengths'].append(summary_length)
        metrics['response_lengths'].append(response_length)
        metrics['total_requests'] += 1
        if error:
            metrics['error_counts'] += 1
            
        # Keep only last 1000 measurements
        for key in ['processing_times', 'sentiment_scores', 'summary_lengths', 'response_lengths']:
            if len(metrics[key]) > 1000:
                metrics[key].pop(0)
    
    def get_variant_summary(self, variant: ExperimentVariant) -> Dict[str, Any]:
        metrics = self.variant_metrics[variant]
        if not metrics['processing_times']:
            return {
                'avg_processing_time': None,
                'error_rate': 0,
                'total_requests': 0,
                'avg_sentiment': None,
                'avg_summary_length': None,
                'avg_response_length': None
            }
            
        return {
            'avg_processing_time': sum(metrics['processing_times']) / len(metrics['processing_times']),
            'error_rate': metrics['error_counts'] / metrics['total_requests'] if metrics['total_requests'] > 0 else 0,
            'total_requests': metrics['total_requests'],
            'avg_sentiment': sum(metrics['sentiment_scores']) / len(metrics['sentiment_scores']),
            'avg_summary_length': sum(metrics['summary_lengths']) / len(metrics['summary_lengths']),
            'avg_response_length': sum(metrics['response_lengths']) / len(metrics['response_lengths'])
        }

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

@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class ModelPipeline:
    def __init__(self):
        logger.info("Initializing ModelPipeline...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.performance_metrics = PerformanceMetrics()
        self.experiment_metrics = ExperimentMetrics()
        
        self.initialize_models()
        self.setup_variants()
        logger.info("ModelPipeline initialization complete")

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

    def initialize_models(self):
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Initialize BERT
            logger.info("Loading BERT...")
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = BertModel.from_pretrained(
                "bert-base-uncased",
                output_attentions=True
            ).to(self.device).eval()

            # Initialize RoBERTa
            logger.info("Loading RoBERTa...")
            self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.roberta_model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels=2
            ).to(self.device).eval()

            # Initialize DistilBERT
            logger.info("Loading DistilBERT...")
            self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.distilbert_model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=2
            ).to(self.device).eval()

            # Initialize GPT-2
            logger.info("Loading GPT-2...")
            self.gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device).eval()

            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def setup_variants(self):
        self.variant_configs = {
            ExperimentVariant.A: {
                'sentiment_model': 'roberta',
                'temperature': 0.7,
                'max_length': 150,
                'num_return_sequences': 1,
                'no_repeat_ngram_size': 2
            },
            ExperimentVariant.B: {
                'sentiment_model': 'distilbert',
                'temperature': 0.9,
                'max_length': 200,
                'num_return_sequences': 1,
                'no_repeat_ngram_size': 3
            }
        }

    def get_variant(self, force_variant: Optional[ExperimentVariant] = None) -> ExperimentVariant:
        return force_variant if force_variant else random.choice([ExperimentVariant.A, ExperimentVariant.B])

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
            # Select model based on type
            if model_type == "roberta":
                tokenizer = self.roberta_tokenizer
                model = self.roberta_model
            else:
                tokenizer = self.distilbert_tokenizer
                model = self.distilbert_model

            # Analyze sentiment
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                return scores[0][1].item()

    async def generate_response(self, text: str, config: Dict[str, Any]) -> Dict[str, str]:
        with timer(self.performance_metrics, "gpt2"):
            # Prepare input
            inputs = self.gpt2_tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.gpt2_model.generate(
                    **inputs,
                    max_length=config['max_length'],
                    num_return_sequences=config['num_return_sequences'],
                    no_repeat_ngram_size=config['no_repeat_ngram_size'],
                    temperature=config['temperature']
                )

            response = self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"generated_response": response}

    async def process_variant(self, text: str, variant: ExperimentVariant) -> Dict[str, Any]:
        """Process text using the specified variant configuration"""
        config = self.variant_configs[variant]
        
        # Get summary
        summary_result = await self.summarize(text)
        summary = summary_result["summary"]
        
        # Get sentiment
        sentiment = await self.analyze_sentiment(text, config['sentiment_model'])
        
        # Generate response
        response_result = await self.generate_response(text, config)
        generated_response = response_result["generated_response"]
        
        result = {
            "variant": variant.value,
            "sentiment_model": config['sentiment_model'],
            "summary": summary,
            "sentiment": sentiment,
            "generated_response": generated_response
        }
        
        return result

    @app.post("/process")
    async def process_text(self, request: TextRequest) -> Dict[str, Any]:
        start_time = time.time()
        request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        try:
            # Determine variant
            variant = self.get_variant(request.force_variant)
            logger.info(f"Processing request {request_id} with variant {variant.value}")
            
            # Get initial system metrics
            initial_metrics = self.get_system_metrics()
            
            # Process with selected variant
            result = await self.process_variant(request.text, variant)
            
            # Calculate processing time and record metrics
            total_time = time.time() - start_time
            self.experiment_metrics.record_metrics(
                variant,
                total_time,
                result["sentiment"],
                len(result["summary"]),
                len(result["generated_response"]),
                error=False
            )
            
            # Add performance data to response
            result["performance_metrics"] = {
                "processing_time": total_time,
                "system_metrics": self.get_system_metrics()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {str(e)}")
            if 'variant' in locals():
                self.experiment_metrics.record_metrics(
                    variant,
                    time.time() - start_time,
                    0.0,
                    0,
                    0,
                    error=True
                )
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    # @app.get("/metrics")
    # async def get_metrics(self) -> Dict[str, Any]:
    #     """Get current performance metrics and experiment statistics"""
    #     try:
    #         # Get current system metrics
    #         current_metrics = self.get_system_metrics()
            
    #         # Get average performance metrics
    #         avg_metrics = {
    #             metric_name: self.performance_metrics.get_average(metric_name)
    #             for metric_name in self.performance_metrics.metrics.keys()
    #         }
            
    #         # Get experiment metrics for both variants
    #         experiment_metrics = {
    #             'variant_A': self.experiment_metrics.get_
    @app.get("/metrics")
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the model service performance and system resources.
        Returns performance metrics for each model, system resource utilization, and experiment metrics.
        """
        try:
            # Get GPU metrics if available
            gpu_metrics = {}
            if torch.cuda.is_available():
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_metrics[f"gpu_{i}"] = {
                        "gpu_utilization": gpu.load * 100,  # Convert to percentage
                        "gpu_memory_used": gpu.memoryUsed,
                        "gpu_memory_total": gpu.memoryTotal,
                        "gpu_temperature": gpu.temperature
                    }

            # Get system metrics
            system_metrics = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used": psutil.virtual_memory().used / (1024 * 1024 * 1024),  # Convert to GB
                "memory_total": psutil.virtual_memory().total / (1024 * 1024 * 1024)  # Convert to GB
            }

            # Calculate model inference metrics
            model_metrics = {
                "bert": {
                    "avg_inference_time": self.performance_metrics.get_average("bert_inference_times"),
                    "num_inferences": len(self.performance_metrics.metrics["bert_inference_times"])
                },
                "roberta": {
                    "avg_inference_time": self.performance_metrics.get_average("roberta_inference_times"),
                    "num_inferences": len(self.performance_metrics.metrics["roberta_inference_times"])
                },
                "distilbert": {
                    "avg_inference_time": self.performance_metrics.get_average("distilbert_inference_times"),
                    "num_inferences": len(self.performance_metrics.metrics["distilbert_inference_times"])
                },
                "gpt2": {
                    "avg_inference_time": self.performance_metrics.get_average("gpt2_inference_times"),
                    "num_inferences": len(self.performance_metrics.metrics["gpt2_inference_times"])
                }
            }

            # Get experiment metrics for both variants
            experiment_metrics = {
                "variant_A": self.experiment_metrics.get_variant_summary(ExperimentVariant.A),
                "variant_B": self.experiment_metrics.get_variant_summary(ExperimentVariant.B)
            }

            # Compile all metrics
            return {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": system_metrics,
                "gpu_metrics": gpu_metrics,
                "model_metrics": model_metrics,
                "experiment_metrics": experiment_metrics
            }

        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error collecting metrics: {str(e)}")

def main():
    try:
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(runtime_env={"pip": [
                "torch",
                "transformers",
                "fastapi",
                "pydantic"
            ]})

        # Start Ray Serve
        serve.start(http_options={"host": "0.0.0.0", "port": 8000})
        
        # Deploy the application
        serve.run(ModelPipeline.bind())
        
        logger.info("Service is running at http://localhost:8000")
        logger.info("Available endpoint: POST /process")
        
        # Keep the service running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down service...")
        serve.shutdown()
    except Exception as e:
        logger.error(f"Service error: {str(e)}")
        serve.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()