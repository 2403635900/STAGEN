# STAGEN: Spatio-Temporal Attention Graph Evolution Network for Temporal Knowledge Graph Reasoning

STAGEN is a novel framework for temporal knowledge graph reasoning that combines dynamic graph attention networks with neural ordinary differential equations to model the continuous evolution of entity and relation representations over time.

**Features:**

* **Dynamic Graph Attention:** Captures structural dependencies with relation-aware attention mechanisms
* **Neural ODEs:** Models continuous temporal dynamics of entity embeddings
* **Temporal Negative Sampling:** Generates temporally valid negative examples
* **Causal Contrastive Learning:** Enhances temporal discrimination of embeddings

**Requirements:**

* Python 3.8+
* PyTorch 1.12+
* torch-scatter
* torchdiffeq

**Usage:**

1.  Prepare your dataset in the `data/` folder following the ICEWS14 format.
2.  Train the model:
    ```bash
    python main.py
    ```

**Key configuration options in `config.py`:**

* `time_windows`: Number of temporal windows
* `hidden_dim`: Embedding dimension
* `time_horizon`: Prediction time horizon
* `tau`: Temperature parameter for contrastive loss

**`stagen/` directory structure:**

├── config.py       # Configuration settings

├── model.py        # Core model architecture

├── trainer.py      # Training logic

├── evaluate.py     # Evaluation metrics

├── util.py         # Data utilities

├── main.py         # Main entry point

└── data/           # Dataset folder

