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

Clone and set up FLStore using a fully automated script:

```bash
git clone https://github.com/SamuelFountain/FLStore
cd FLStore
bash run_example.sh  # Sets up Conda, OpenFaaS, MinIO, downloads data, deploys functions, and runs experiments interactively
```

This script will:
- Install Conda, OpenFaaS, MinIO, and other dependencies (Docker and containerd should be pre-installed).
- Build and deploy experimental data to MinIO.
- Prompt you to select non-training workloads (functions) such as scheduling, debugging, clustering, etc.
- Run selected workloads for a few rounds, print results, and teardown the dev environment.

---

### 🧹 Cleanup

Use the provided script to clean up resources created by FLStore (especially if using `k3s` and OpenFaaS setup):

```bash
bash cleanup.sh
```

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
├── cleanup.sh             # Cleans up resources created by FLStore deployment (including k3s setup)
├── run_example.sh         # Automated setup and execution of FLStore and experiments
├── set_conda.sh           # Optional manual Conda environment setup script
├── README.md              # Project documentation
├── .gitattributes         # Git attributes configuration
└── .gitignore             # Git ignore configuration
```

---

## 📌 Hardware and Software Dependencies

- **Python:** Listed in `environment.yml` and `requirements.txt`
- **Hardware:** Standard server hardware (no specialized equipment required; GPU optional)
- **Software:**
  - Python (via Anaconda)
  - Docker (must be pre-installed)
  - Containerd (recommended)
  - OpenFaaS
  - MinIO or similar object store (for persistent FL metadata storage)

---

## 📈 Benchmarks

| Metric            | Average Reduction | Peak Reduction |
|-------------------|-------------------|----------------|
| Latency           | 71%               | 99.7%          |
| Operational Cost  | 92.45%            | 98.8%          |

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
