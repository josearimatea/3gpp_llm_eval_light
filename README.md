# 3GPP LLM Evaluation (Lightweight Version) - Reproduction Guide

This repository provides the essential files to reproduce the key findings from the paper

"Lightweight LLMs for 3GPP Specifications: Fine-Tuning, Retrieval-Augmented Generation and Quantization" by José de Arimatéa Passos Lopes Júnior et al. This lightweight version is designed for users to replicate the experiments efficiently, assuming access to modest hardware (e.g., a consumer-grade GPU like NVIDIA RTX 3050 Ti with ~4GB memory). For the full project, including additional experiments and results, refer to the comprehensive repository:

[https://github.com/josearimatea/3gpp_llm_evaluation](https://github.com/josearimatea/3gpp_llm_evaluation).

Or see branch "full" updated for this project.

The steps below guide you through cloning the repository, setting up the environment, preparing the data, fine-tuning the model, integrating RAG, and generating the accuracy results as presented in the paper.

---

## Prerequisites

Before starting, ensure you have the following:

- **Hardware**: A machine with a GPU (e.g., NVIDIA RTX 3050 Ti, ~4GB VRAM) or access to Google Colab with a GPU (e.g., Tesla T4).

- **Software**:

- Python 3.8+

- Git

- pip (Python package manager)

- CUDA (if using a local GPU; compatible with your GPU, e.g., CUDA 11.8)

- **Dependencies**: Install required Python packages (see Step 2).

- **Disk Space**: ~20-30 GB for datasets, models, and embeddings.

---

## Step-by-Step Instructions

### 1. Clone the Repository

Clone this lightweight repository to your local machine or Colab environment:

```bash

git clone https://github.com/josearimatea/3gpp_llm_eval_light.git

cd 3gpp_llm_eval_light

```

This repository contains the essential scripts and configuration files to replicate the experiments.

---

### 2. Set Up the Environment

Install the necessary Python dependencies:

```bash

pip install torch transformers sentence-transformers faiss-gpu unsloth langchain pandas numpy jupyter

```

**Notes**:

- If using a local GPU, ensure `torch` and `faiss-gpu` are installed with CUDA support matching your hardware.

- For Colab, these packages are typically pre-installed or can be installed via `pip install` in a notebook cell.

---

### 3. Download and Prepare Datasets

The experiments rely on two datasets: **TeleQnA** and **TSpec-LLM**.

#### **TeleQnA Dataset**

- **Source**: [https://github.com/netop-team/TeleQnA](https://github.com/netop-team/TeleQnA)

- **Action**: Clone or download the dataset, then process it using the Jupyter notebook:

```bash

jupyter notebook  # Open `Process_TeleQnA.ipynb` and execute all cells

```

This script prepares a subset of 4,000 questions for fine-tuning and 600 for testing.

#### **TSpec-LLM Dataset**

- **Source**: [https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM](https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM)

- **Action**: Download via Hugging Face CLI:

```bash

pip install datasets

python -c "from datasets import load_dataset; load_dataset('rasoul-nikbakht/TSpec-LLM', split='train').to_json('Tspec-LLM.json')"

```

Process the data using:

```bash

jupyter notebook  # Open `Process_tspec_llm.ipynb` and execute all cells

```

This generates ~780,651 chunks (size=2000, overlap=100) for RAG.

**Storage**: Place the processed files in a `data/` directory within the repository.

---

### 4. Fine-Tune the Llama 3.2 3B Model

Fine-tune the model using the Unsloth framework on 4,000 TeleQnA questions, using Google Colab if you don't have more than 7GB GPU memory:

```bash

jupyter notebook  # Open `Llama_fine_tunning_4000_short_answer_label_colab.ipynb` and run all cells

```

**Parameters**:

- Model: `unsloth/LLama-3.2-3B-Instruct` (unquantized for training)

- Batch Size: 32 (2 per device, 16 gradient accumulation steps)

- Steps: 400 (~3.2 epochs)

- Learning Rate: 0.0002

- Optimizer: AdamW (8-bit precision)

**Output**:

- Fine-tuned model (`Llama-4bit-Tuned`) saved and quantized to 4 bits (~2.768 GB).

**Note**: Training requires ~6.688 GB VRAM (unquantized).

---

### 5. Set Up Retrieval-Augmented Generation (RAG)

Integrate RAG using TSpec-LLM chunks:

- **Embedding Model**: `SentenceTransformer('all-mpnet-base-v2')`

- **Script**: Run chunking and indexing (if not done in Step 3):

```bash

jupyter notebook  # Open `Process_tspec_llm.ipynb` and execute

```

**Parameters**:

- Chunk Size: 2000 characters

- Overlap: 100 characters

- Top-k: 5 chunks retrieved (via Faiss with cosine similarity)

**Output**: Indexed embeddings stored locally (~4-hour process on RTX 3050 Ti).

---

### 6. Run Inference and Evaluate Models

Test models with and without RAG on 600 TeleQnA questions.

#### Scripts:

- **Without RAG**:

- `Inference_gpt4_mini.ipynb`

- `Inference_llama_3.2.ipynb`

- `Inference_llama_3.2_lora_short_answer.ipynb`

- **With RAG**:

- `Inference_RAG_gpt4_mini.ipynb`

- `Inference_RAG_llama_3.2.ipynb`

- `Inference_RAG_llama_3.2_lora.ipynb`

**Execution**:

```bash

jupyter notebook  # Open each script, adjust paths, and run all cells

```

**Prompts**:

- **No RAG**: `"Question: {question}\nOptions: {options}\nThink step-by-step and respond with 'correct option: X'"`

- **With RAG**: `"Relevant Information: {rag_results}\nQuestion: {question}\nOptions: {options}\nThink step-by-step and respond with 'correct option: X'"`

**Models**:

- **GPT-4o-mini**: Requires OpenAI API key.

- **Llama 3.2 3B**: Quantized to 4 bits.

- **Llama-4bit-Tuned**: Fine-tuned model from Step 4.

---

### 7. Generate Results

Compile accuracy results using:

```bash

jupyter notebook  # Open `main_results_600_questions.ipynb`, adjust paths, and run all cells

```

**Output**:

- **GPT-4o-mini**: RAG 74.0%, No RAG 64.3%

- **Llama-4bit-Tuned**: RAG 76.3%, No RAG 61.0%

- **Llama 3.2 3B**: RAG 67.8%, No RAG 54.7%

---

## Troubleshooting

- **Memory Issues**: Reduce batch size or use Colab if GPU memory exceeds 3.712 GB.

- **API Errors**: Ensure a valid OpenAI API key for GPT-4o-mini.

- **File Paths**: Update paths in scripts to match your local `data/` and `model/` directories.

---

## References

- Paper: *"Lightweight LLMs for 3GPP Specifications: Fine-Tuning, Retrieval-Augmented Generation and Quantization"*

- Full Repository: [https://github.com/josearimatea/3gpp_llm_evaluation](https://github.com/josearimatea/3gpp_llm_evaluation)

- Datasets: [TeleQnA](https://github.com/netop-team/TeleQnA), [TSpec-LLM](https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM)

- Unsloth: [https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)

