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
import time
import logging
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

@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class ModelPipeline:
    def __init__(self):
        logger.info("Initializing ModelPipeline...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.initialize_models()
        self.setup_variants()
        logger.info("ModelPipeline initialization complete")

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
        # Tokenize and prepare input
        inputs = self.bert_tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        # Generate summary
        with torch.no_grad():
            outputs = self.bert_model(**inputs, output_attentions=True)
            attention = outputs.attentions[-1].mean(dim=1)
            scores = attention.mean(dim=-1).squeeze()

        # Extract best sentence
        sentences = text.split('.')
        if len(sentences) > 1 and len(scores) > 0:
            best_idx = min(int(scores.argmax()), len(sentences) - 1)
            summary = sentences[best_idx].strip()
        else:
            summary = sentences[0].strip()

        return {"summary": summary}

    async def analyze_sentiment(self, text: str, model_type: str) -> float:
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

    @app.post("/process")
    async def process_text(self, request: TextRequest) -> Dict[str, Any]:
        start_time = time.time()
        variant = self.get_variant(request.force_variant)
        config = self.variant_configs[variant]

        try:
            # Process the text
            summary_result = await self.summarize(request.text)
            sentiment = await self.analyze_sentiment(request.text, config['sentiment_model'])
            response_result = await self.generate_response(request.text, config)

            # Prepare response
            result = {
                "variant": variant.value,
                "summary": summary_result["summary"],
                "sentiment": sentiment,
                "generated_response": response_result["generated_response"],
                "processing_time": time.time() - start_time
            }

            return result

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

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