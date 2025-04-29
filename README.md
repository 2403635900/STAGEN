# SIGMOD26-STAGEN: Decoupled Causal Reasoning in Temporal Knowledge Graphs

STAGEN is a novel framework for temporal knowledge graph reasoning that combines dynamic graph attention networks with neural ordinary differential equations to model the continuous evolution of entity and relation representations over time.

**Abstract:**
 Temporal knowledge graph reasoning faces critical challenges in
 modeling complex spatio-temporal interactions and resisting tem
poral distribution shifts. We propose STAGEN (Spatio-Temporal
 Augmented Graph Evolution Network), a novel architecture that
 innovatively integrates multiscale ODEs with causal contrastive
 learning. First, we decouple the dynamics modeling into spatial
 topology evolution and temporal feature propagation through a
 dual-stream ODE structure, enabling precise capture of both lo
cal structural changes and global pattern transitions. Second, we
 design a dynamic relation-aware graph convolutional layer com
bined with multi-head attention to effectively model time-varying
 relation dependencies. Third, we introduce a causal intervention
 mechanism in counterfactual contrastive learning to enhance the
 model’s robustness against temporal perturbations. Experiments
 on three benchmark datasets demonstrate state-of-the-art perfor
manceofourmodel,withanMRRimprovementof11.03%compared
 to baseline methods.

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

## Dataset
There are five datasets (from [RE-NET](https://github.com/INK-USC/RE-Net)): ICEWS18, ICEWS14, GDELT, WIKI, and YAGO. Times of test set should be larger than times of train and valid sets. (Times of valid set also should be larger than times of train set.) Each data folder has 'stat.txt', 'train.txt', 'valid.txt', 'test.txt'.

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

