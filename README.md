# 📚 ESE 577 Final Project — Domain-Specific Chatbot with Mistral-7B

This repository contains the code, data, and documentation for our final project in **ESE 577: Deep Learning — Algorithms and Software** at Stony Brook University. We fine-tuned the **Mistral-7B-Instruct-v0.1** model to build a chatbot that can answer questions specifically about the ESE 577 course content.

## 👨‍🏫 Course Information

- **Course**: ESE 577 – Deep Learning Algorithms and Software  
- **Semester**: Fall 2024  
- **Instructor**: Jorge Mendez-Mendez  
- **Team Members**: Ningyuan Yang, Yiti Li, Guoao Li  
- **Emails**: `{ningyuan.yang, yiti.li, guoao.li}@stonybrook.edu`

---

## 🎯 Project Goal

The objective is to **fine-tune a large language model (LLM)** using LoRA and quantization techniques to act as a chatbot specialized in answering questions based on the ESE 577 course syllabus. This chatbot can help students clarify course content, deadlines, grading policies, and lecture topics.

---

## 🧠 Model and Approach

- **Base Model**: `Mistral-7B-Instruct-v0.1`  
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)  
- **Precision**: 4-bit quantized weights (QLoRA)  
- **Tokenization**: Hugging Face tokenizer  
- **Training Framework**: PyTorch + Hugging Face Transformers

---

## 📁 File Structure
.
├── code/
│ ├── ese577_fall24_project.ipynb
├── data/
│ ├── ESE_577_syllabus.pdf
│ ├── QA_pair.xlsx
│ ├── validation_set.xlsx
│ └── testing_set.xlxs
├── ESE_577_Deep_Learning_Project_Spring_2025.pdf
├── ESE_577_project.pdf
└── README.md

---

## 🗃️ Dataset

- **Training Set**: 100 QA pairs manually generated from the ESE 577 syllabus
- **Validation Set**: 12 multiple-choice QA items
- **Test Set**: 20 paraphrased QA pairs for qualitative and quantitative evaluation

---

## ⚙️ Training Configuration

- **Epochs**: 20
- **Batch Size**: 4
- **Learning Rate**: 2e-4 (AdamW)
- **LoRA Rank**: 64
- **LoRA Alpha**: 16
- **Dropout**: 0.1
- **Sequence Length**: 256 tokens
- **Loss Function**: Cross-entropy
- **Validation Strategy**: Best model by lowest validation loss (early stopping at epoch 3)

---

## 📊 Evaluation Metrics

| Metric        | Before Fine-tuning | After Fine-tuning |
|---------------|--------------------|-------------------|
| BLEU          | 0.0598             | 0.1127            |
| ROUGE-1       | 0.2050             | 0.2923            |
| ROUGE-2       | 0.0985             | 0.1747            |
| ROUGE-L       | 0.1785             | 0.2481            |

---

## 🔍 Sample QA Results

| Question | Before Fine-tuning | After Fine-tuning |
|----------|--------------------|-------------------|
| *How many quizzes are counted?* | Random guess | Correct: "10 best quizzes" |
| *When is gradient descent covered?* | Irrelevant text | Correct date: 09/05 |
| *Academic integrity site?* | Incorrect URLs | Accurate SBU link |

---

## 🚧 Limitations & Future Work

- **Small dataset** → Expand via augmentation or external sources  
- **Model selection** → Add BLEU/ROUGE as validation criteria  
- **Hyperparameter tuning** → Experiment with epochs, dropout, batch size

---

## 🧾 References

- [LoRA: Low-Rank Adaptation of LLMs](https://arxiv.org/abs/2106.09685)  
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)  
- [Mistral-7B Model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)  
- [BLEU and ROUGE Scores](https://www.aclweb.org/anthology/W04-1013/)

---

## 📎 Acknowledgment

All team members contributed equally to data collection, modeling, experimentation, and writing.

---

## 💻 How to Run

1. Clone the repo and open `ese577_fall24_project.ipynb` in Google Colab.
2. Ensure model and tokenizer are available via Hugging Face.
3. Run training cells and use `generate()` to test chatbot behavior.
4. Evaluate using QA pairs from `validation_set.xlsx`.

---
