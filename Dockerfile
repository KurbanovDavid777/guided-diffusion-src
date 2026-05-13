# Guided Diffusion for Next-in-Class Molecule Generation
# Docker image with both environments: small_molecules + targetdiff

FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    wget curl git \
    libxrender1 libxext6 \
    openbabel \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /workspace

# ── Python dependencies ────────────────────────────────────────────────────

# small_molecules dependencies
RUN pip install --no-cache-dir \
    rdkit \
    biopython \
    requests \
    pandas \
    numpy \
    tqdm \
    plip

# targetdiff dependencies
RUN pip install --no-cache-dir \
    torch-geometric==2.6.1 \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    scikit-learn \
    scipy \
    easydict \
    pyyaml

# AutoDock Vina
RUN pip install --no-cache-dir vina

# ── Copy project code ──────────────────────────────────────────────────────
COPY small_molecules/ /workspace/small_molecules/
COPY targetdiff/ /workspace/targetdiff/

# ── Download pretrained models ─────────────────────────────────────────────
RUN python -c "\
import requests, tarfile, os; \
url = 'https://zenodo.org/api/records/14041881/files/targetdiff_pretrained_models.tar.gz/content'; \
r = requests.get(url, stream=True, allow_redirects=True); \
open('/tmp/models.tar.gz', 'wb').write(r.content); \
tarfile.open('/tmp/models.tar.gz').extractall('/workspace/targetdiff/'); \
os.remove('/tmp/models.tar.gz'); \
print('Models downloaded!')"

# ── Environment variables ──────────────────────────────────────────────────
ENV PYTHONPATH=/workspace/targetdiff:$PYTHONPATH

# ── Default command ────────────────────────────────────────────────────────
CMD ["/bin/bash"]
