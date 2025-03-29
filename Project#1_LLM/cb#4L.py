#!/usr/bin/env python
"""
Rewritten Chatbot with Persistent and Adaptive Memory
-------------------------------------------------------
This version refactors the original code into modular sections:
  - Global configuration and imports
  - Helper functions for embedding (with normalization), memory scoring/pruning, and context summarization
  - Memory modules: PersistentMemory, TitanMemory, SimpleAttention, TitanArchitecture
  - Chatbot class: handles LLM interaction, tool command processing, and memory management

Tool execution outputs (“products”) are consistently saved in memory.
Note: All memory operations now use the automatically returned "ids" field 
      (the "uris" field is always None in your current ChromaDB version).
"""

import os
import regex as re  # note the change from 'import re'
import time
import json
import uuid
import math
import datetime
import asyncio
import warnings
import logging
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import spacy
from transformers import pipeline

import numpy as np

# External tool module (must be available in your environment)
from tool_module import ToolModule

# ------------------ Global Settings and Cache ------------------
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("chatbot_titan.log"), logging.StreamHandler()]
)

FILE_CACHE = {}  # For caching file contents

CONFIG = {
    "MAX_CONTEXT": int(os.getenv("MAX_CONTEXT", 10)),
    "MAX_MEMORIES": int(os.getenv("MAX_MEMORIES", 1000)),
    "MEMORY_SCORING": {
        "weight_sim": 1.0,
        "weight_recency": 0.5,
        "weight_importance": 0.2,
        "time_decay_constant": 3600  # seconds
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using PyTorch device: {device}")

# ------------------ Initialize External Dependencies ------------------
try:
    chroma_client = chromadb.PersistentClient(path="./memory_db")
    # When adding memories we use "ids". Retrieval, update, and deletion will use these same IDs.
    memory_collection = chroma_client.get_or_create_collection(
        name="chat_memory",
        embedding_function=None
    )
    logging.info("✅ ChromaDB collection 'chat_memory' initialized successfully.")
except Exception as e:
    logging.critical(f"❌ Failed to initialize ChromaDB collection: {e}")
    raise

try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_model.to(device)
    logging.info("✅ SentenceTransformer model loaded successfully.")
except Exception as e:
    logging.critical(f"❌ Failed to load SentenceTransformer model: {e}")
    raise

try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("✅ spaCy NER model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Could not load spaCy NER model: {e}")
    nlp = None

try:
    summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn", 
        device=0 if torch.cuda.is_available() else -1
    )
    logging.info("✅ Summarization model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load summarization model: {e}")
    summarizer = None

# ------------------ Helper Functions ------------------
def flatten_list(lst):
    """Recursively flatten a nested list."""
    if not lst:
        return []
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat

def embed_text(text: str) -> list:
    """
    Converts text to an embedding vector.
    The embedding is normalized (L2 norm) to improve similarity calculations.
    (Optionally, dimensionality reduction techniques such as PCA could be applied here.)
    """
    if not isinstance(text, str):
        text = json.dumps(text)
    try:
        # Generate tensor embedding and normalize it.
        embedding_tensor = embed_model.encode(text, convert_to_tensor=True).to(device)
        embedding_tensor = F.normalize(embedding_tensor, p=2, dim=0)
        embedding = embedding_tensor.flatten().tolist()
        return embedding
    except Exception as e:
        logging.error(f"⚠️ Embedding error: {e}")
        return []

def compute_memory_score(similarity: float, memory_meta: dict, current_time: datetime.datetime,
                         scoring_config: dict = CONFIG["MEMORY_SCORING"]) -> float:
    try:
        timestamp_str = memory_meta.get("timestamp")
        if timestamp_str:
            timestamp = datetime.datetime.fromisoformat(timestamp_str)
            delta = (current_time - timestamp).total_seconds()
            recency_score = math.exp(-delta / scoring_config["time_decay_constant"])
        else:
            recency_score = 0.5
    except Exception:
        recency_score = 0.5
    recalled_count = memory_meta.get("recalled_count", 0)
    importance_score = math.log(1 + recalled_count)
    return (scoring_config["weight_sim"] * similarity +
            scoring_config["weight_recency"] * recency_score +
            scoring_config["weight_importance"] * importance_score)

def prune_memory(max_entries: int = CONFIG["MAX_MEMORIES"]) -> None:
    try:
        # Retrieve all memories (we request "metadatas"; "ids" is returned automatically)
        all_data = memory_collection.get(include=["metadatas"])
        id_list = flatten_list(all_data.get("ids", []))
        metadatas = flatten_list(all_data.get("metadatas", []))
        total_memories = len(id_list)
        if total_memories > max_entries * 1.5:
            def sort_key(item):
                _, meta = item
                return (meta.get("recalled_count", 0), meta.get("timestamp", ""))
            sorted_entries = sorted(zip(id_list, metadatas), key=sort_key, reverse=True)
            to_delete = [mem_id for mem_id, _ in sorted_entries[total_memories - max_entries:]]
            memory_collection.delete(ids=to_delete)
            logging.info(f"Pruned {len(to_delete)} memories; total was {total_memories}.")
    except Exception as e:
        logging.error(f"Error pruning memory: {e}")

def summarize_context(chat_history: list) -> str:
    if not summarizer:
        logging.warning("Summarization model not available.")
        return "[Context summary unavailable]"
    try:
        messages = [msg["content"] for msg in chat_history if msg["role"] != "system"]
        full_text = " ".join(messages)[:1000]
        max_length = 150 if len(full_text) >= 100 else 50
        summary_result = summarizer(full_text, max_length=max_length, min_length=40, do_sample=False)
        summary_text = summary_result[0]['summary_text']
        return f"[Summary: {summary_text}]"
    except Exception as e:
        logging.error(f"Error summarizing context: {e}")
        return "[Context summary error]"

def save_memory(role: str, content: str) -> None:
    if not content:
        return
    if not isinstance(content, str):
        content = json.dumps(content)
    unique_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    emb = embed_text(content)
    if not emb or not all(isinstance(x, (float, int)) for x in emb):
        logging.error("🔴 Invalid embedding format; skipping memory save.")
        return
    logging.info(f"📌 Saving memory [{role}]: {content[:50]}... (ID={unique_id})")
    try:
        memory_collection.add(
            documents=[content],
            metadatas=[{"role": role, "timestamp": timestamp, "recalled_count": 0}],
            ids=[unique_id],  # Using "ids" when adding a memory.
            embeddings=[emb]
        )
        logging.info(f"✅ Memory saved with ID {unique_id}")
    except Exception as e:
        logging.error(f"🔴 Failed to save memory: {e}")

def cleanup_invalid_ids() -> None:
    try:
        # Retrieve memories; "ids" is automatically returned.
        all_data = memory_collection.get(include=["metadatas"])
        id_list = flatten_list(all_data.get("ids", []))
        for mem_id in id_list:
            if not mem_id or not isinstance(mem_id, str):
                logging.warning(f"Invalid memory ID found: {mem_id}")
                memory_collection.delete(ids=[mem_id])
    except Exception as e:
        logging.error(f"Error during invalid ID cleanup: {e}")

def get_relevant_memories(query: str, top_k: int = 7) -> list:
    query_emb = embed_text(query)
    if not query_emb:
        return []
    
    # Display stored metadata for debugging.
    try:
        all_metadatas = memory_collection.get(include=["metadatas"])["metadatas"]
        print("Stored Metadata:", all_metadatas)
    except Exception as e:
        logging.error(f"Failed to retrieve stored metadata: {e}")
    
    # Adjust retrieval threshold using memory_collection.count()
    try:
        total_memories = memory_collection.count()
    except Exception as e:
        logging.error(f"Failed to count memories: {e}")
        total_memories = 0
    n_results = min(top_k * 5, total_memories) if total_memories > 0 else top_k

    try:
        np_query_emb = np.array(query_emb)
        results = memory_collection.query(
            query_embeddings=[np_query_emb],
            n_results=n_results,
            include=["documents", "metadatas", "uris", "distances"]  # "ids" is returned automatically.
        )
    except Exception as e:
        logging.error(f"🔴 ChromaDB query failed: {e}")
        return []
    
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    # Instead of using "uris" (which is None), we extract IDs from "ids"
    id_list = flatten_list(results.get("ids", []))
    distances = flatten_list(results.get("distances", []))
    if not (docs and metas and id_list and distances):
        return []
    
    meta_list = flatten_list(metas)
    doc_list = flatten_list(docs)
    dist_list = flatten_list(distances)
    candidates = []
    now = datetime.datetime.now()
    for mem_id, doc_text, meta, dist in zip(id_list, doc_list, meta_list, dist_list):
        if not mem_id or not isinstance(mem_id, str):
            logging.warning(f"Skipping memory with invalid ID: {mem_id}")
            continue
        similarity = 1 - dist  # Convert distance to similarity.
        score = compute_memory_score(similarity, meta, now)
        # Prioritize assistant responses by boosting their scores.
        if meta.get("role") == "assistant":
            score *= 1.5
        candidates.append({
            "id": mem_id,
            "role": meta.get("role", "user"),
            "content": doc_text,
            "timestamp": meta.get("timestamp"),
            "recalled_count": meta.get("recalled_count", 0),
            "similarity": similarity,
            "score": score,
            "meta": meta
        })
    
    sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    retrieved = []
    for candidate in sorted_candidates[:top_k * 2]:
        meta = candidate["meta"]
        meta["recalled_count"] = meta.get("recalled_count", 0) + 1
        try:
            # Use "ids" in the update call.
            memory_collection.update(ids=[candidate["id"]], metadatas=[meta])
        except Exception as update_error:
            logging.warning(f"Could not update recalled_count for {candidate['id']}: {update_error}")
        candidate["recalled_count"] = meta["recalled_count"]
        retrieved.append({
            "id": candidate["id"],
            "role": candidate["role"],
            "content": candidate["content"],
            "timestamp": candidate["timestamp"],
            "recalled_count": candidate["recalled_count"]
        })
    
    # Similarity check for debugging.
    from sklearn.metrics.pairwise import cosine_similarity
    try:
        query_emb_arr = np.array(embed_text(query)).reshape(1, -1)
        all_memories = memory_collection.get(include=["documents"])
        stored_docs = all_memories.get("documents", [])
        stored_embs = np.array([embed_text(doc) for doc in stored_docs])
        if stored_embs.size > 0:
            similarities = cosine_similarity(query_emb_arr, stored_embs)
            print("Query Similarities:", similarities)
        else:
            print("No stored embeddings available for similarity comparison.")
    except Exception as e:
        logging.error(f"Error computing cosine similarities: {e}")
    
    logging.info(f"🔎 Retrieved {len(retrieved)} relevant memories.")
    return retrieved

# ------------------ Memory Modules ------------------
class PersistentMemory(nn.Module):
    def __init__(self, mem_dim: int = CONFIG["TITAN_PARAMS"]["VALUE_DIM"]):
        super().__init__()
        self.knowledge_vector = nn.Parameter(torch.randn(mem_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.knowledge_vector

class TitanMemory(nn.Module):
    def __init__(self, input_dim: int, key_dim: int, value_dim: int, hidden_dim: int, params: dict):
        super().__init__()
        self.eta = params["eta"]
        self.theta = params["theta"]
        self.alpha = params["alpha"]
        self.use_directional_surprise = params["use_directional_surprise"]

        self.W_K = nn.Linear(input_dim, key_dim, bias=False)
        self.W_V = nn.Linear(input_dim, value_dim, bias=False)
        self.memory_net = nn.Sequential(
            nn.Linear(key_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, value_dim)
        )
        self.momentum_buffers = {
            name: torch.zeros_like(param.data).to(device)
            for name, param in self.memory_net.named_parameters()
        }

    def update_memory(self, x: torch.Tensor):
        k = self.W_K(x)
        v = self.W_V(x)
        pred = self.memory_net(k)
        loss = F.mse_loss(pred, v)
        grads = torch.autograd.grad(loss, self.memory_net.parameters(), create_graph=False)
        if self.use_directional_surprise:
            total_sim, count = 0, 0
            for g in grads:
                if g is not None:
                    total_sim += (g.view(-1) ** 2).sum()
                    count += 1
            avg_surprise = torch.sqrt(total_sim / count) if count > 0 else torch.tensor(0.0, device=device)
        else:
            grad_norms = [g.norm() for g in grads if g is not None]
            avg_surprise = sum(grad_norms) / len(grad_norms) if grad_norms else torch.tensor(0.0, device=device)
        effective_theta = self.theta * torch.sigmoid(avg_surprise)
        for (name, param), grad in zip(self.memory_net.named_parameters(), grads):
            buf = self.momentum_buffers[name]
            new_buf = self.eta * buf - effective_theta * grad
            param.data = (1 - self.alpha) * param.data + new_buf
            self.momentum_buffers[name] = new_buf.detach()
        logging.debug(f"TitanMemory updated: loss={loss.item():.4f}, avg_surprise={avg_surprise.item():.4f}")
        return loss.item(), pred.detach(), v.detach()

    def get_memory_context(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            k = self.W_K(x)
            return self.memory_net(k)

class SimpleAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query_vector: torch.Tensor, history_vectors: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(query_vector)
        k = self.key_proj(history_vectors)
        attn_scores = torch.matmul(q, k.T)
        attn_weights = F.softmax(attn_scores, dim=1)
        return torch.matmul(attn_weights, history_vectors)

class TitanArchitecture(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self.persistent_mem = PersistentMemory(mem_dim=params["VALUE_DIM"]).to(device)
        self.titan_memory = TitanMemory(
            input_dim=params["INPUT_DIM"],
            key_dim=params["KEY_DIM"],
            value_dim=params["VALUE_DIM"],
            hidden_dim=params["HIDDEN_DIM"],
            params=params
        ).to(device)
        self.short_term_attn = SimpleAttention(params["INPUT_DIM"]).to(device)
        self.short_term_net = nn.Linear(params["INPUT_DIM"], params["VALUE_DIM"]).to(device)
        self.gate_net = nn.Sequential(nn.Linear(params["VALUE_DIM"], 1), nn.Sigmoid()).to(device)
        self.gate_target_net = nn.Sequential(nn.Linear(params["VALUE_DIM"], 1), nn.Sigmoid()).to(device)
        self.input_proj = nn.Linear(params["INPUT_DIM"], params["VALUE_DIM"]).to(device)
        self.entity_proj = nn.Linear(params["INPUT_DIM"], params["VALUE_DIM"]).to(device)
        self.entity_gate = nn.Sequential(nn.Linear(params["VALUE_DIM"], 1), nn.Sigmoid()).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.short_term_attn.parameters()) +
            list(self.short_term_net.parameters()) +
            list(self.gate_net.parameters()) +
            list(self.gate_target_net.parameters()) +
            list(self.input_proj.parameters()) +
            list(self.entity_proj.parameters()) +
            list(self.entity_gate.parameters()) +
            list(self.persistent_mem.parameters()),
            lr=1e-3
        )
        self.grad_clip_max_norm = 1.0

    def forward(self, current_input: torch.Tensor,
                history_embeddings: torch.Tensor,
                entity_embeddings: torch.Tensor = None):
        titan_loss, _, _ = self.titan_memory.update_memory(current_input)
        titan_context = self.titan_memory.get_memory_context(current_input)
        persistent_out = self.persistent_mem(titan_context)
        if history_embeddings is not None and history_embeddings.size(0) > 0:
            short_term_vec = self.short_term_attn(current_input, history_embeddings)
            short_term_context = self.short_term_net(short_term_vec)
        else:
            short_term_context = torch.zeros((1, persistent_out.size(1))).to(device)
        gate_val = self.gate_net(persistent_out)
        predicted_gate = self.gate_target_net(persistent_out)
        proj_input = self.input_proj(current_input)
        ltm_sim = (F.cosine_similarity(proj_input, persistent_out, dim=1) + 1) / 2
        stm_sim = (F.cosine_similarity(proj_input, short_term_context, dim=1) + 1) / 2
        ideal_gate = (ltm_sim / (ltm_sim + stm_sim + 1e-8)).detach().unsqueeze(1)
        gating_loss = F.mse_loss(predicted_gate, ideal_gate)
        self.optimizer.zero_grad()
        gating_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            list(self.short_term_attn.parameters()) +
            list(self.gate_net.parameters()) +
            list(self.short_term_net.parameters()) +
            list(self.gate_target_net.parameters()) +
            list(self.input_proj.parameters()) +
            list(self.entity_proj.parameters()) +
            list(self.entity_gate.parameters()),
            self.grad_clip_max_norm
        )
        self.optimizer.step()
        fused = gate_val * persistent_out + (1 - gate_val) * short_term_context
        if entity_embeddings is not None and entity_embeddings.size(0) > 0:
            entity_avg = torch.mean(entity_embeddings, dim=0, keepdim=True)
            entity_context = self.entity_proj(entity_avg)
            entity_gate_val = self.entity_gate(entity_context)
            fused = entity_gate_val * entity_context + (1 - entity_gate_val) * fused
        return fused, titan_loss, gate_val.item(), predicted_gate.item(), gating_loss

# ------------------ Chatbot Class ------------------
class Chatbot:
    def __init__(self):
        params = CONFIG["TITAN_PARAMS"]
        self.titan_arch = TitanArchitecture(params).to(device)
        if os.path.exists("titan_memory.pt"):
            try:
                self.titan_arch.titan_memory.load_state_dict(
                    torch.load("titan_memory.pt", map_location=device)
                )
                logging.info("🔄 Loaded TitanMemory from 'titan_memory.pt'")
            except Exception as e:
                logging.error(f"Failed to load TitanMemory: {e}")
        if os.path.exists("persistent_mem.pt"):
            try:
                self.titan_arch.persistent_mem.load_state_dict(
                    torch.load("persistent_mem.pt", map_location=device)
                )
                logging.info("🔄 Loaded PersistentMemory from 'persistent_mem.pt'")
            except Exception as e:
                logging.error(f"Failed to load PersistentMemory: {e}")
        self.tool = ToolModule()
        self.entity_memory = {}

    def parse_tool_command(self, response_text: str) -> dict:
        try:
            cleaned_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
            cleaned_text = re.sub(r"<!--.*?-->", "", cleaned_text, flags=re.DOTALL)
            start = cleaned_text.find("{")
            end = cleaned_text.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = cleaned_text[start:end].strip()
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing tool command: {e}")
        return None

    def execute_tool_command(self, cmd: dict) -> str:
        action = cmd.get("action")
        action_input = cmd.get("action_input")
        logging.info(f"⚙️ Executing tool command: action={action}, input={action_input}")
        tool_result = ""
        try:
            if action == "search_web":
                tool_result = self.tool.search_web(action_input)
                print("Search Results:", tool_result)
            elif action == "read_file":
                if action_input in FILE_CACHE:
                    tool_result = FILE_CACHE[action_input]
                    logging.info(f"✅ Using cached file content for: {action_input}")
                else:
                    tool_result = self.tool.read_file(action_input)
                    FILE_CACHE[action_input] = tool_result
                print("File Content:", tool_result)
            elif action == "alter_file":
                parts = [p.strip() for p in action_input.split(",")]
                if len(parts) == 3:
                    logging.info(f"Altering file with parameters: {parts}")
                    self.tool.alter_file(*parts)
                    tool_result = f"Successfully altered {parts[0]}"
                    print(tool_result)
                else:
                    tool_result = ("Error: Invalid format for alter_file. Expected: "
                                   "<file_path>, <search_text>, <replace_text>")
                    print(tool_result)
            else:
                tool_result = f"Error: Unknown action '{action}'."
                print(tool_result)
        except Exception as e:
            logging.error(f"Error during tool execution: {e}")
            tool_result = f"Tool execution failed: {e}"
        save_memory("tool", str(tool_result))
        logging.info(f"✅ Tool execution completed with result: {tool_result}")
        return tool_result

    def process_llm_response(self, chat_history: list) -> str:
        max_retries = 3
        ai_response = ""
        for attempt in range(max_retries):
            try:
                response = ollama.chat(model="huihui_ai/deepseek-r1-abliterated", messages=chat_history)
                ai_response = response['message']['content']
                break
            except Exception as e:
                logging.error(f"Error in Ollama chat (attempt {attempt+1}): {e}")
                time.sleep(1)
                ai_response = "I'm sorry, I encountered an error."
        while True:
            cmd = self.parse_tool_command(ai_response)
            if cmd is not None and "action" in cmd:
                tool_output = self.execute_tool_command(cmd)
                print(f"[Tool Output]: {tool_output}")
                chat_history.append({"role": "tool", "content": str(tool_output)})
                save_memory("tool", str(tool_output))
                for attempt in range(max_retries):
                    try:
                        response = ollama.chat(model="huihui_ai/deepseek-r1-abliterated", messages=chat_history)
                        ai_response = response['message']['content']
                        break
                    except Exception as e:
                        logging.error(f"Error in Ollama chat (attempt {attempt+1}): {e}")
                        time.sleep(1)
                        ai_response = "I'm sorry, I encountered an error."
                logging.info(f"LLM response after tool call: {ai_response}")
            else:
                break
        return ai_response

    def run(self) -> None:
        system_prompt = {
            "role": "system",
            "content": (
                "You are RazerBlade, a private, uncensored friend with memory, dedicated to providing accurate, relevant, and context-aware responses. "
                "David is your friend. Always maintain clarity, recall user context precisely, and ensure your responses are grounded in verifiable data.\n\n"
                "IMPORTANT GUIDELINES:\n"
                "1️⃣ When a tool command is required, output ONLY a valid JSON object with NO additional text, explanation, or internal markers. "
                "The JSON must exactly match this format:\n"
                "   { \"action\": \"tool_name\", \"action_input\": \"input_for_the_tool\" }\n"
                "   Do NOT include any internal reasoning markers (such as <think>...</think>) or extra commentary.\n"
                "2️⃣ You may only use these tools:\n"
                "   - search_web (for current online information)\n"
                "   - read_file (for retrieving file content)\n"
                "   - alter_file (for modifying file text)\n"
                "3️⃣ If no tool is needed, respond in plain text with no JSON formatting.\n\n"
                "EXAMPLES:\n"
                "✅ Correct: { \"action\": \"search_web\", \"action_input\": \"Gas stations near 5465 N. Shelton Ave., Wichita KS 67204\" }\n"
                "❌ Incorrect: { \"action\": \"gas_station_finder\", \"action_input\": \"5465 N. Shelton Ave., Wichita KS 67204\" }\n\n"
                "Your final responses must be based on verified data and must not include any extraneous text when a tool command is required."
            )
        }
        
        while True:
            try:
                user_input = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                logging.info("User terminated the session. Saving memories...")
                torch.save(self.titan_arch.titan_memory.state_dict(), "titan_memory.pt")
                torch.save(self.titan_arch.persistent_mem.state_dict(), "persistent_mem.pt")
                print("AI: Goodbye! Memories saved.")
                break

            retrieved_memories = get_relevant_memories(user_input)
            sorted_memories = sorted(retrieved_memories, key=lambda m: m.get("timestamp", ""))
            for mem in sorted_memories:
                if "address" in user_input.lower() and "address" in mem["content"].lower():
                    print(f"AI: {mem['content']}")
                    save_memory("assistant", mem["content"])
                    prune_memory()
                    continue

            chat_history = [system_prompt] + [{"role": m["role"], "content": m["content"]} for m in sorted_memories]
            chat_history.append({"role": "user", "content": user_input})
            context_summary = summarize_context(chat_history)
            chat_history.insert(1, {"role": "assistant", "content": context_summary})
            if user_input.startswith("!"):
                continue

            entity_emb_list = []
            doc = None
            if nlp:
                try:
                    doc = nlp(user_input)
                    filtered_entities = [(ent.text, ent.label_) for ent in doc.ents if ".txt" not in ent.text]
                    for ent in doc.ents:
                        try:
                            ent_emb = embed_model.encode(ent.text, convert_to_tensor=True).unsqueeze(0).to(device)
                            entity_emb_list.append(ent_emb)
                            if ent.text in self.entity_memory:
                                self.entity_memory[ent.text] = (self.entity_memory[ent.text] + ent_emb) / 2
                            else:
                                self.entity_memory[ent.text] = ent_emb
                        except Exception as e:
                            logging.error(f"Error encoding entity '{ent.text}': {e}")
                    if filtered_entities:
                        entity_str = "; ".join([f"{txt}({lab})" for txt, lab in filtered_entities])
                        chat_history.insert(2, {"role": "assistant", "content": f"[Recognized Entities: {entity_str}]"})
                except Exception as e:
                    logging.error(f"Error during NER processing: {e}")

            user_emb_list = []
            for msg in chat_history:
                if msg["role"].lower() == "user":
                    try:
                        emb_tensor = embed_model.encode(msg["content"], convert_to_tensor=True).unsqueeze(0).to(device)
                        user_emb_list.append(emb_tensor)
                    except Exception as e:
                        logging.error(f"Error encoding message: {e}")
            hist_emb = torch.cat(user_emb_list, dim=0) if user_emb_list else None

            try:
                cur_emb = embed_model.encode(user_input, convert_to_tensor=True).unsqueeze(0).to(device)
                cur_emb.requires_grad = True
            except Exception as e:
                logging.error(f"Error encoding current input: {e}")
                continue

            entity_emb_tensor = torch.cat(entity_emb_list, dim=0) if entity_emb_list else None

            fused_out, titan_loss, gate_val, predicted_gate, gating_loss = self.titan_arch(
                cur_emb, hist_emb, entity_emb_tensor
            )
            logging.info(f"TitanMemory: loss={titan_loss:.4f}, gate={gate_val:.3f}, predicted_gate={predicted_gate:.3f}, gating_loss={gating_loss.item():.4f}")

            if nlp and doc and hist_emb is not None:
                for ent in doc.ents:
                    try:
                        ent_emb = embed_model.encode(ent.text, convert_to_tensor=True).unsqueeze(0).to(device)
                        _, ent_loss, ent_gate_val, ent_predicted_gate, _ = self.titan_arch(ent_emb, hist_emb)
                        logging.info(f"Entity '{ent.text}' ({ent.label_}) update: loss={ent_loss:.4f}, gate={ent_gate_val:.3f}")
                    except Exception as e:
                        logging.error(f"Entity update error for '{ent.text}': {e}")
            ai_response = self.process_llm_response(chat_history)
            print(f"AI: {ai_response}")
            save_memory("user", user_input)
            save_memory("assistant", ai_response)
            prune_memory()

# ------------------ Demo Tool Usage (Optional) ------------------
def demo_tool_usage(tool: ToolModule) -> None:
    print("=== Demonstrating ToolModule Usage ===")
    results = tool.search_web("Python ChatGPT tutorial", max_results=3)
    print("\n[Demo] Search Results:", results)
    file_content = tool.read_file("notes.txt")
    print("\n[Demo] File Content of 'notes.txt':", file_content)
    tool.write_file("notes.txt", "New line from demo!", mode='a')
    print("[Demo] Appended a new line to 'notes.txt'")
    tool.alter_file("notes.txt", "old phrase", "new phrase")
    print("[Demo] Replaced 'old phrase' with 'new phrase' in 'notes.txt'")
    article_text = tool.extract_article("https://example.com")
    print("\n[Demo] Extracted Article Text (first 300 chars):", article_text[:300], "...")
    token_count = tool.count_tokens(article_text)
    print("\n[Demo] Token count of article:", token_count)
    
    async def do_async_searches():
        queries = ["Python threading", "Best AI frameworks", "SpaceX news"]
        tasks = [tool.search_web_async(q) for q in queries]
        return await asyncio.gather(*tasks)
    
    loop = asyncio.get_event_loop()
    async_results = loop.run_until_complete(do_async_searches())
    print("\n[Demo] Async search results:", async_results)
    print("=== End Demo ===")

# ------------------ Main Execution ------------------
if __name__ == "__main__":
    cleanup_invalid_ids()
    chatbot = Chatbot()
    chatbot.run()

