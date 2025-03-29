#!/usr/bin/env python
import os
import warnings
import logging
import torch
import chromadb
from sentence_transformers import SentenceTransformer
import spacy
from transformers import pipeline

# ------------------ Global Cache ------------------
FILE_CACHE = {}

# ------------------ Configuration ------------------
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("chatbot_titan.log"), logging.StreamHandler()]
)

CONFIG = {
    "MAX_CONTEXT": int(os.getenv("MAX_CONTEXT", 10)),
    "MAX_MEMORIES": int(os.getenv("MAX_MEMORIES", 1000)),
    "MEMORY_SCORING": {
        "weight_sim": 1.0,
        "weight_recency": 0.5,
        "weight_importance": 0.2,
        "time_decay_constant": 3600
    },
    "TITAN_PARAMS": {
        "eta": 0.9,
        "theta": 0.1,
        "alpha": 0.1,
        "use_directional_surprise": True,
        "INPUT_DIM": 384,
        "VALUE_DIM": 128,
        "KEY_DIM": 128,
        "HIDDEN_DIM": 256
    }
}

# ------------------ Initialize Core Dependencies ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using PyTorch device: {device}")

# Initialize ChromaDB collection
try:
    client = chromadb.PersistentClient(path="./memory_db")
    collection = client.get_or_create_collection(
        name="chat_memory",
        embedding_function=None
    )
    logging.info("✅ ChromaDB collection 'chat_memory' initialized successfully.")
except Exception as e:
    logging.critical(f"❌ Failed to initialize ChromaDB collection: {e}")
    raise

# Initialize SentenceTransformer
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_model.to(device)
    logging.info("✅ SentenceTransformer model loaded successfully.")
except Exception as e:
    logging.critical(f"❌ Failed to load SentenceTransformer model: {e}")
    raise

# Initialize spaCy (for named-entity recognition)
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("✅ spaCy NER model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Could not load spaCy NER model: {e}")
    nlp = None

# Initialize Summarization Pipeline
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    logging.info("✅ Summarization model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load summarization model: {e}")
    summarizer = None

