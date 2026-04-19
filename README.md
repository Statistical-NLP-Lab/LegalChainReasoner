# LegalChainReasoner

This repository contains the code and data for the paper **"LegalChainReasoner: Grounding Criminal Judicial Opinion Generation via Structured Legal Chains"**.

LegalChainReasoner is a framework designed to unify **Legal Reasoning** and **Sentencing Prediction** into a single generation process — **Criminal Judicial Opinion Generation (CJOG)**. Statutory provisions are formalized as structured **Legal Chains** (Premise → Situation → Conclusion triplets) and injected into an LLM through a **Chain-Aware Encoding** module, so that sentencing decisions are grounded in and justified by the same legal analysis that produces the reasoning text.

---

## 📂 Repository Structure

```
.
├── LegalChainReasoner.py        # Main training and evaluation script (Source code)
├── requirements.txt
├── README.md
├── data/
│   ├── LAIC/                    # Main data from LAIC-2021
│   ├── PCCD/                    # Zero-shot evaluation data from PCCD
│   └── CAIL/                    # Zero-shot evaluation data from CAIL-2018
└── legal_chain/                 # Structured Legal Chains
    ├── robbery/                   
    │── fraud/                   
    └── ...
```

---

## 🛠 Requirements

The project is implemented using PyTorch and HuggingFace Transformers. To set up the environment, please run:

```
pip install -r requirements.txt
```

This demo code is built upon Llama-3.2-3B. Ensure you have access to HuggingFace Hub.

**Key Dependencies:**

- Python >= 3.8
- PyTorch (CUDA support needed)
- Transformers
- Peft (for LoRA fine-tuning)
- Thulac (for Chinese word segmentation)
- Rouge (for evaluation)

---

## 📊 Datasets

We utilize three datasets for experiments:

1. **LAIC-2021:** Derived from the Legal AI Challenge 2021, containing cases with factual descriptions, reasoning, and sentencing outcomes.
2. **PCCD:** A curated set of complex cases from the People’s Court Case Database for zero-shot generalization evaluation.
3. **CAIL-2018:** A subset of CAIL2018, used zero-shot for sentencing prediction only. 

The processed datasets are located in the `data/` directory. Each line in the data files is a JSON object mainly containing:

| Field | Description |
|---|---|
| `caseCause` | The crime type (e.g., "robbery"). |
| `justice`   | The case facts (Fact). |
| `opinion`   | The court's reasoning (Reasoning). |
| `judge`     | The prison term in months (Sentencing). |

---

## 🔗 Legal Chains

For each crime type we provide three files:

| File                 | Purpose                                                      |
| -------------------- | ------------------------------------------------------------ |
| `chain.txt`          | Legal Chain in the form `Premise -> Situation -> Conclusion`. |
| `<crime>_nodes.json` | Node-level metadata.                                         |
| `<crime>.txt`        | Edge list describing the chain graph.                        |

---

## 📥 Requesting Full Data & Resources

To keep this release compact, the following resources are demonstrated here and are available to researchers upon request:

- Full **LAIC-2021** training set
- Full curated **PCCD** evaluation set
- Full curated **CAIL2018** evaluation set
- Full **Legal Chains** covering all crime categories used in the paper

To request access, please contact the authors.

---

## 🚀 Usage

### 1. Preparation

Ensure that the Legal Chain folder contains the extracted chain files for the relevant crimes. The model uses these external knowledge bases to construct the Chain-Aware embeddings.

### 2. Training & Evaluation

The main script handles both training (with LoRA) and evaluation. To start the training process:

```
python LegalChainReasoner.py
```

### 3. Configuration

The training hyperparameters and file paths are currently configured within the main script. To change settings (e.g., batch size, learning rate, or device), please modify the `if __name__ == '__main__':` block inside `LegalChainReasoner.py`:

```
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Set GPU device
    epochs = 100                    # Number of training epochs
    batch_size = 1                  # Training batch size
    
    # ... ensure the file paths below match your data directory
    train = open("./data/LAIC/train_data.json", 'r', encoding='utf-8').readlines()
    test = open("./data/LAIC/test_data.json", 'r', encoding='utf-8').readlines()
```

Note that the script ships with the MAE / RMSE / ROUGE evaluation loop. The additional metrics reported in the paper (BLEU-1/2/N, BERTScore, GPTScore-pairwise, Legal-QA) are computed with the standard external tools described in the paper's Appendix.
