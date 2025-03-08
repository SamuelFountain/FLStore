# 🚀 FLStore: Efficient Federated Learning Storage for Non-Training Workloads

**FLStore** is an innovative **serverless framework** designed to optimize the storage and execution of **non-training workloads** in **Federated Learning (FL)**.

FLStore addresses challenges of **non-training workloads**—such as scheduling, debugging, personalization, and clustering—by dramatically reducing **latency** and **operational costs** through serverless caching and tailored data storage.

✅ **Performance Highlights:**
- **Latency:** Average reduction of **71%** (peak **99.7%**)
- **Cost:** Average savings of **92.45%** (peak **98.8%**)

FLStore integrates seamlessly into existing FL frameworks with minimal changes.

---

## 📦 Quick Installation

### Quick Setup (Linux)

Clone and set up FLStore quickly:

```bash
git clone https://github.com/SamuelFountain/FLStore
cd FLStore
bash set_conda.sh  # Customize if needed
pip install -r requirements.txt && pip install -e .
```

**Tip:** For GPU support or custom parameters, update `set_conda.sh`.

---

### Installation from Source (Linux/MacOS)

With Anaconda installed, run:

```bash
cd FLStore

# Set FLStore home
conda init bash
export FLSTORE_HOME=$(pwd)
echo 'export FLSTORE_HOME=$(pwd)' >> ~/.bashrc
source ~/.bashrc

# Create and activate conda environment
conda env create -f environment.yml
conda activate flstore
pip install -r requirements.txt && pip install -e .
```

**Note:** Install Docker and OpenFaaS if deploying serverless functions in production.

---

### 📂 Repository Structure

```plaintext
FLStore/
├── fetch_experiments/    # Scripts to fetch and post FL metadata
│   ├── __init__.py       # Python package initialization
│   ├── fetch.py          # Data fetching script
│   └── post.py           # Data posting script
├── run_experiments/      # Experimental evaluation scripts
│   ├── __init__.py       # Python package initialization
│   ├── experiment1.py    # First experiment script
│   └── experiment2.py    # Second experiment script
├── serverless/           # Serverless caching and compute functions
│   ├── __init__.py       # Python package initialization
│   ├── cache.py          # Caching functionality
│   └── compute.py        # Computation functionality
├── cleanup.sh            # Script to clean resources
├── run_example.sh        # Basic FLStore deployment example script
├── set_conda.sh          # Conda environment setup script
├── README.md             # Documentation
├── .gitattributes        # Git attributes configuration
└── .gitignore            # Git ignore configuration
```

---

## 📌 Hardware and Software Dependencies

- **Python:** Listed in `environment.yml` and `requirements.txt`
- **Hardware:** Standard server hardware (no specialized equipment required; GPU optional)
- **Software:**
  - Python (via Anaconda)
  - Docker and OpenFaaS
  - MinIO or similar object store (for persistent FL metadata storage)

---

## 📈 Benchmarks

| Metric           | Average Reduction | Peak Reduction |
|------------------|-------------------|----------------|
| Latency          | 71%               | 99.7%          |
| Operational Cost | 92.45%            | 98.8%          |

---

## 📖 Citation

Please cite the following if using FLStore:

```bibtex
@misc{khan2025flstore,
  author = {Khan, et al.},
  title = {FLStore: Efficient Federated Learning Storage},
  year = {2025},
  eprint = {2503.00323},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2503.00323}
}
```

---

## 🤝 Contributions and Communication

Contributions welcome! Submit bugs, features, or improvements via GitHub issues or pull requests.
