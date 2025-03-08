# 🚀 FLStore: Efficient Federated Learning Storage for Non-Training Workloads

**FLStore** is an innovative **serverless framework** designed to optimize the storage and execution of **non-training workloads** in **Federated Learning (FL)**.

FLStore addresses challenges of **non-training workloads**—such as scheduling, debugging, personalization, and clustering—by dramatically reducing **latency** and **operational costs** through serverless caching and tailored data storage.

✅ **Performance Highlights:**
- **Latency:** Average reduction of **71%** (peak **99.7%**)
- **Cost:** Average savings of **92.45%** (peak **98.8%**)

FLStore integrates seamlessly into existing FL frameworks with minimal modifications.

---

## 📦 Quick Installation

### Quick Setup (Recommended: Ubuntu Linux)

Clone and quickly set up FLStore using the provided script, which fully automates environment setup, dependency installation, MinIO, OpenFaaS, and experiment execution:

```bash
git clone https://github.com/SamuelFountain/FLStore
cd FLStore
bash run_example.sh  # Automatically sets up environment, installs MinIO, OpenFaaS, and all dependencies
```

**Tip:** Customize `run_example.sh` for GPU support or other specific requirements.

---

### 📂 Repository Structure

```plaintext
FLStore/
├── fetch_experiments/     # Scripts to fetch and post FL metadata
│   ├── __init__.py        # Python package initialization
│   ├── fetch.py           # Data fetching script
│   └── post.py            # Data posting script
├── run_experiments/       # Experimental evaluation scripts
│   ├── __init__.py        # Python package initialization
│   ├── experiment1.py     # First experiment script
│   └── experiment2.py     # Second experiment script
├── serverless/            # Serverless caching and compute functions
│   ├── __init__.py        # Python package initialization
│   ├── cache.py           # Caching functionality
│   └── compute.py         # Computation functionality
├── cleanup.sh             # Cleans up resources created by FLStore deployment
├── run_example.sh         # Fully automated FLStore setup and experiment execution script
├── set_conda.sh           # Conda environment setup (optional manual usage)
├── README.md             # Project documentation
├── .gitattributes         # Git attributes configuration
└── .gitignore             # Git ignore configuration
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

Comparisons made against traditional cloud object storage (e.g., AWS S3) and cloud caching solutions (e.g., AWS ElastiCache).

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
