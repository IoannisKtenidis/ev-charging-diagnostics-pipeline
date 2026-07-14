import os
import json
import random
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocpp_pipeline")

# ==========================================
# 1. DEPENDENCY FALLBACK HANDLING
# ==========================================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not found. Pipeline will run using mock/simulated inference.")

try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.warning("MLflow not found. Pipeline will skip MLflow tracking.")

try:
    import bentoml
    HAS_BENTOML = True
except ImportError:
    HAS_BENTOML = False
    logger.warning("BentoML not found. BentoML service decorator will be bypassed.")

# ==========================================
# 2. OCPP LOG PARSER & VOCABULARY
# ==========================================
VOCAB = {
    "PAD": 0,
    "UNK": 1,
    "MASK": 2,
    "BootNotification": 3,
    "Heartbeat": 4,
    "StatusNotification": 5,
    "Authorize": 6,
    "TransactionEvent": 7,
    "StartTransaction": 8,
    "StopTransaction": 9,
    "MeterValues": 10,
    "Response": 11,
    "Error": 12
}
REV_VOCAB = {v: k for k, v in VOCAB.items()}

class OCPPLogParser:
    """Parses raw JSON OCPP frames into categorical log template IDs."""
    
    @staticmethod
    def parse_message(raw_msg: str) -> str:
        """Extracts the template action from a raw OCPP message."""
        try:
            parsed = json.loads(raw_msg)
            if not isinstance(parsed, list) or len(parsed) < 2:
                return "UNK"
            
            msg_type = parsed[0]
            if msg_type == 2: # Call
                return parsed[2] if len(parsed) > 2 else "UNK"
            elif msg_type == 3: # CallResult
                return "Response"
            elif msg_type == 4: # CallError
                return "Error"
            return "UNK"
        except Exception:
            return "UNK"

    @classmethod
    def to_sequence_ids(cls, raw_messages: list[str], max_len: int = 10) -> list[int]:
        """Converts a sequence of raw messages to a padded list of token IDs."""
        ids = []
        for msg in raw_messages:
            template = cls.parse_message(msg)
            ids.append(VOCAB.get(template, VOCAB["UNK"]))
        
        # Padding
        if len(ids) < max_len:
            ids += [VOCAB["PAD"]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids

# ==========================================
# 3. UNSUPERVISED PyTorch MLM DATASET
# ==========================================
if HAS_TORCH:
    class OCPPMLMDataset(Dataset):
        """Unsupervised MLM Dataset that masks 15% of tokens for LogBERT training."""
        def __init__(self, sequences, mask_prob=0.15):
            self.sequences = sequences
            self.mask_prob = mask_prob

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            seq = torch.tensor(self.sequences[idx], dtype=torch.long)
            target = seq.clone()
            
            # Mask 15% of tokens (excluding PAD=0, UNK=1, MASK=2)
            rand = torch.rand(seq.shape)
            mask_arr = (rand < self.mask_prob) & (seq > 2)
            
            # Loss is only calculated for masked tokens (ignore index = -100)
            target[~mask_arr] = -100
            
            # Replace masked tokens with MASK token (ID=2)
            seq[mask_arr] = 2
            
            return seq, target
else:
    class OCPPMLMDataset:
        def __init__(self, *args, **kwargs): pass

# ==========================================
# 4. LogBERT MODEL DEFINITION (PYTORCH)
# ==========================================
if HAS_TORCH:
    class LogBERT(nn.Module):
        """Transformer-based LogBERT model for MLM log sequence representation."""
        def __init__(self, vocab_size=len(VOCAB), embed_dim=32, num_heads=2, num_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_embedding = nn.Parameter(torch.randn(1, 20, embed_dim))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=64, 
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(embed_dim, vocab_size)

        def forward(self, x):
            seq_len = x.size(1)
            emb = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
            out = self.transformer(emb)
            logits = self.fc(out)
            return logits
else:
    class LogBERT:
        def __init__(self, *args, **kwargs): pass

# ==========================================
# 5. DATA INGESTION & PIPELINE ENGINE
# ==========================================
class DiagnosticsPipeline:
    def __init__(self):
        self.model = None
        self.anomaly_threshold = 0.35
        self.weights_path = os.path.join(os.path.dirname(__file__), "logbert_weights.pth")
        self.init_model()

    def init_model(self):
        """Initializes PyTorch model and loads saved weights if present."""
        if HAS_TORCH:
            self.model = LogBERT()
            if os.path.exists(self.weights_path):
                try:
                    self.model.load_state_dict(torch.load(self.weights_path, map_location=torch.device('cpu')))
                    logger.info("Loaded pre-trained LogBERT weights successfully.")
                except Exception as e:
                    logger.warning(f"Could not load pre-trained weights: {str(e)}. Initializing clean weights.")
            self.model.eval()
        else:
            self.model = "MOCK_MODEL"

    def load_sequences(self, dataset_path: str, seq_len: int = 10) -> list[list[int]]:
        """Loads and parses raw OCPP logs, grouping them into sliding window sequences per stationId."""
        if not os.path.exists(dataset_path):
            logger.warning(f"Log dataset path {dataset_path} not found. Returning empty list.")
            return []
            
        station_messages = {}
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    sid = data.get("stationId", "default")
                    msg = data.get("message", "")
                    if sid not in station_messages:
                        station_messages[sid] = []
                    station_messages[sid].append(msg)
                except Exception:
                    continue
                    
        all_sequences = []
        for sid, messages in station_messages.items():
            ids = [VOCAB.get(OCPPLogParser.parse_message(m), VOCAB["UNK"]) for m in messages]
            # Construct sliding window sequences with step size 3 to capture sequence transitions
            for i in range(0, len(ids) - seq_len + 1, 3):
                all_sequences.append(ids[i : i + seq_len])
                
        logger.info(f"Ingested and parsed {len(all_sequences)} sequences of length {seq_len} from {dataset_path}.")
        return all_sequences

    def train_unsupervised(self, dataset_path: str, epochs: int = 10, batch_size: int = 32):
        """Trains LogBERT on normal sequences using Unsupervised MLM, tracked by MLflow."""
        if not HAS_TORCH:
            logger.error("PyTorch is required for model training.")
            return
            
        sequences = self.load_sequences(dataset_path)
        if not sequences:
            logger.warning("No sequences loaded. Training canceled.")
            return
            
        # Split sequences into train (80%) and validation (20%) datasets
        shuffled_seqs = list(sequences)
        random.seed(42)  # For reproducibility
        random.shuffle(shuffled_seqs)
        
        split_idx = int(len(shuffled_seqs) * 0.8)
        train_seqs = shuffled_seqs[:split_idx]
        val_seqs = shuffled_seqs[split_idx:]
        
        train_dataset = OCPPMLMDataset(train_seqs)
        val_dataset = OCPPMLMDataset(val_seqs)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.model = LogBERT()
        self.model.train()
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # Cosine Annealing learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        best_val_loss = float("inf")
        patience = 3
        patience_counter = 0
        
        if HAS_MLFLOW:
            mlflow.set_experiment("OCPP_Unsupervised_LogBERT")
            with mlflow.start_run() as run:
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("vocab_size", len(VOCAB))
                mlflow.log_param("dataset_size", len(sequences))
                mlflow.log_param("train_size", len(train_seqs))
                mlflow.log_param("val_size", len(val_seqs))
                mlflow.log_param("optimizer", "Adam")
                mlflow.log_param("initial_lr", 0.001)
                mlflow.log_param("patience", patience)
                
                mlflow.set_tag("model_type", "LogBERT")
                mlflow.set_tag("task", "Unsupervised MLM Anomaly Detection")
                
                for epoch in range(epochs):
                    # --- TRAINING ---
                    self.model.train()
                    train_loss = 0.0
                    train_correct = 0
                    train_total = 0
                    
                    for seqs, targets in train_loader:
                        if not (targets != -100).any():
                            continue
                        optimizer.zero_grad()
                        outputs = self.model(seqs)
                        
                        loss = criterion(outputs.view(-1, len(VOCAB)), targets.view(-1))
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item() * seqs.size(0)
                        
                        preds = torch.argmax(outputs, dim=-1)
                        mask = (targets != -100)
                        train_correct += (preds[mask] == targets[mask]).sum().item()
                        train_total += mask.sum().item()
                        
                    avg_train_loss = train_loss / len(train_seqs) if train_seqs else 0.0
                    train_acc = train_correct / train_total if train_total > 0 else 0.0
                    
                    # --- VALIDATION ---
                    self.model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for seqs, targets in val_loader:
                            if not (targets != -100).any():
                                continue
                            outputs = self.model(seqs)
                            loss = criterion(outputs.view(-1, len(VOCAB)), targets.view(-1))
                            val_loss += loss.item() * seqs.size(0)
                            
                            preds = torch.argmax(outputs, dim=-1)
                            mask = (targets != -100)
                            val_correct += (preds[mask] == targets[mask]).sum().item()
                            val_total += mask.sum().item()
                            
                    avg_val_loss = val_loss / len(val_seqs) if val_seqs else 0.0
                    val_acc = val_correct / val_total if val_total > 0 else 0.0
                    
                    current_lr = optimizer.param_groups[0]['lr']
                    scheduler.step()
                    
                    # Log metrics to MLflow
                    mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                    mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                    mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                    mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                    mlflow.log_metric("learning_rate", current_lr, step=epoch)
                    
                    logger.info(
                        f"Epoch {epoch+1:02d}/{epochs:02d} | "
                        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                        f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%} | "
                        f"LR: {current_lr:.6f}"
                    )
                    
                    # Early stopping check
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        torch.save(self.model.state_dict(), self.weights_path)
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping triggered at epoch {epoch+1} (Validation loss didn't improve for {patience} epochs)")
                            mlflow.set_tag("early_stopping_epoch", epoch+1)
                            break
                
                # Load best weights before logging model
                if os.path.exists(self.weights_path):
                    self.model.load_state_dict(torch.load(self.weights_path))
                mlflow.pytorch.log_model(self.model, "logbert_ocpp_unsupervised", serialization_format="pickle")
                logger.info(f"Unsupervised training complete. Model logged to MLflow under run {run.info.run_id}")
        else:
            # Train without MLflow if not installed
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                for seqs, targets in train_loader:
                    if not (targets != -100).any():
                        continue
                    optimizer.zero_grad()
                    outputs = self.model(seqs)
                    loss = criterion(outputs.view(-1, len(VOCAB)), targets.view(-1))
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * seqs.size(0)
                    
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for seqs, targets in val_loader:
                        if not (targets != -100).any():
                            continue
                        outputs = self.model(seqs)
                        loss = criterion(outputs.view(-1, len(VOCAB)), targets.view(-1))
                        val_loss += loss.item() * seqs.size(0)
                        
                avg_train_loss = train_loss / len(train_seqs) if train_seqs else 0.0
                avg_val_loss = val_loss / len(val_seqs) if val_seqs else 0.0
                scheduler.step()
                
                logger.info(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), self.weights_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
            if os.path.exists(self.weights_path):
                self.model.load_state_dict(torch.load(self.weights_path))
        
        self.model.eval()

    def evaluate_sequence(self, raw_messages: list[str]) -> dict:
        """Evaluates sequence probability using the trained LogBERT model."""
        templates = [OCPPLogParser.parse_message(msg) for msg in raw_messages]
        token_ids = OCPPLogParser.to_sequence_ids(raw_messages)
        
        probabilities = []
        is_anomaly = False
        
        if HAS_TORCH and isinstance(self.model, nn.Module):
            with torch.no_grad():
                input_tensor = torch.tensor([token_ids], dtype=torch.long)
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=-1)[0]
                
                # Extract predicted Softmax probability of each actual token in sequence
                for i, token_id in enumerate(token_ids):
                    if token_id == VOCAB["PAD"]:
                        continue
                    prob = float(probs[i, token_id])
                    probabilities.append(prob)
        else:
            # Heuristics fallback
            for token_id in token_ids:
                if token_id == VOCAB["PAD"]:
                    continue
                if token_id in (VOCAB["Error"], VOCAB["UNK"]):
                    prob = 0.05
                else:
                    prob = 0.85
                probabilities.append(prob)
                
        avg_prob = sum(probabilities) / len(probabilities) if probabilities else 1.0
        
        # Flag anomaly if threshold breached or critical Error frame matched
        if "Error" in templates or avg_prob < self.anomaly_threshold:
            is_anomaly = True
            
        return {
            "timestamp": datetime.now().isoformat(),
            "sequence": templates,
            "token_ids": token_ids,
            "probabilities": [round(p, 4) for p in probabilities],
            "average_probability": round(avg_prob, 4),
            "is_anomaly": is_anomaly
        }

# ==========================================
# 6. BentoML SERVICE DEFINITION
# ==========================================
pipeline = DiagnosticsPipeline()

if HAS_BENTOML:
    @bentoml.service(
        name="ocpp_logbert_service",
        resources={"cpu": "200m"}
    )
    class OCPPLogBERTService:
        def __init__(self):
            self.pipeline = pipeline

        @bentoml.api
        async def evaluate(self, sequence: list[str]) -> dict:
            """Serves the sequence evaluations via REST API."""
            try:
                return self.pipeline.evaluate_sequence(sequence)
            except Exception as e:
                logger.error(f"Error in evaluate: {str(e)}")
                return {"error": str(e), "is_anomaly": True}
else:
    class OCPPLogBERTService:
        def __init__(self):
            self.pipeline = pipeline
        async def evaluate(self, sequence: list[str]) -> dict:
            return self.pipeline.evaluate_sequence(sequence)

if __name__ == "__main__":
    # If run directly, run test check on local dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    path1 = os.path.join(base_dir, "LOGS", "ocpp-normal-dataset.jsonl")
    path2 = os.path.join(base_dir, "logs", "ocpp-normal-dataset.jsonl")
    dataset = path1 if os.path.exists(path1) else path2
    
    if os.path.exists(dataset):
        logger.info(f"Found local normal dataset at {dataset}. Testing pipeline load:")
        seqs = pipeline.load_sequences(dataset)
        print(f"Total window sequences loaded: {len(seqs)}")
    else:
        logger.warning("No normal dataset found to load.")
