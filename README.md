# DermaAI: Dermatology Image Classifier with Clinical NLP Dashboard

> **Built an end-to-end Healthcare AI platform combining transfer-learned ResNet-18, Grad-CAM explainability, spaCy NER, DistilBERT severity classification, and a fully interactive Streamlit dashboard for automated dermatology analysis and clinical report generation.**

---

# 🏥 Project Overview

DermaAI is a full-stack Healthcare AI system that combines **Computer Vision + NLP + Explainable AI** into a unified clinical workflow.

The platform:

1. **Classifies** dermoscopic skin lesion images into 7 diagnostic categories using a fine-tuned ResNet-18 CNN.
2. **Visualizes** model attention using Grad-CAM heatmaps for explainability.
3. **Extracts** structured medical entities from patient symptom reports using spaCy NER.
4. **Predicts** symptom severity using a fine-tuned DistilBERT transformer.
5. **Generates** structured clinical draft reports using template-based NLG.
6. **Provides** a modern interactive Streamlit dashboard for end-to-end inference and visualization.

---

# 🌐 Streamlit Dashboard Features

### Interactive UI Modules

* 🖼️ Skin lesion image classification
* 🔥 Grad-CAM heatmap visualization
* 📝 Patient symptom report input
* 🔍 Named Entity Recognition (NER)
* ⚖️ Severity prediction (mild/moderate/severe)
* 📋 Automated clinical report generation
* 📊 Evaluation metrics & architecture walkthrough
* 🌙 Modern dark-mode medical dashboard

---

# 📁 Project Structure

```bash
derma_nlp_project/
├── src/
│   ├── dl/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── gradcam.py
│   │
│   ├── nlp/
│   │   ├── data_gen.py
│   │   ├── ner_pipeline.py
│   │   ├── severity_model.py
│   │   └── nlg_module.py
│   │
│   └── utils/
│       ├── config.py
│       └── logger.py
│
├── data/
├── models/
├── outputs/
├── reports/
├── notebooks/
│   └── EDA_and_Demo.ipynb
│
├── tests/
│   ├── test_dataset.py
│   ├── test_ner.py
│   └── test_nlg.py
│
├── Streamlit.py          # Full interactive dashboard
├── demo.py
├── requirements.txt
└── README.md
```

---

# 🚀 Quick Start

## 1️⃣ Clone Repository

```bash
git clone https://github.com/ojeshwigautam/Bias-Lens-Pro.git
cd Bias-Lens-Pro
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 3️⃣ Download HAM10000 Dataset

Download from Kaggle:

```bash
kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection
unzip skin-lesion-analysis-toward-melanoma-detection.zip -d data/
```

---

## 4️⃣ Train CNN Model

```bash
python src/dl/train.py --epochs 15 --batch_size 32 --lr 0.01
```

---

## 5️⃣ Train NLP Models

```bash
python src/nlp/data_gen.py
python src/nlp/severity_model.py
```

---

## 6️⃣ Launch Streamlit Dashboard

```bash
streamlit run Streamlit.py
```

Open browser:

```bash
http://localhost:8501
```

---

# 🧠 Deep Learning Pipeline (CSR311)

| Component         | Detail                                        |
| ----------------- | --------------------------------------------- |
| Dataset           | HAM10000 (10,015 images, 7 classes)           |
| Backbone          | ResNet-18 pretrained on ImageNet              |
| Transfer Learning | Freeze early layers + fine-tune deeper layers |
| Loss              | Weighted CrossEntropyLoss                     |
| Optimizer         | SGD + Momentum                                |
| LR Scheduler      | ReduceLROnPlateau                             |
| Explainability    | Grad-CAM                                      |
| Augmentation      | Flip, Rotate, ColorJitter                     |
| Metrics           | Accuracy, Macro-F1, Recall                    |

---

# 📝 NLP Pipeline (CSR322)

| Component      | Detail                           |
| -------------- | -------------------------------- |
| Dataset        | Synthetic dermatology reports    |
| NER            | spaCy EntityRuler                |
| Entities       | body_part, lesion_type, duration |
| Severity Model | DistilBERT                       |
| Classes        | mild / moderate / severe         |
| NLG            | Template-based report generation |
| Metrics        | Entity F1, BLEU Score            |

---

# 📊 Results

| Metric                           | Score |
| -------------------------------- | ----- |
| CNN Accuracy                     | ~82%  |
| CNN Macro-F1                     | ~0.77 |
| NER F1                           | ~0.88 |
| Severity Classification Accuracy | ~84%  |
| BLEU Score                       | ~0.62 |

---

# 🔥 Grad-CAM Explainability

The system generates Grad-CAM activation maps to visualize which lesion regions influenced CNN predictions.

This improves:

* Model interpretability
* Clinical trust
* AI transparency in healthcare

---

# 🛠️ Tech Stack

### Deep Learning

* PyTorch
* TorchVision
* ResNet-18
* Grad-CAM

### NLP

* HuggingFace Transformers
* DistilBERT
* spaCy
* NLTK

### Frontend

* Streamlit
* Custom CSS UI
* Interactive dashboard components

### Utilities

* scikit-learn
* pandas
* matplotlib

---

# 📄 License

MIT License — educational and research use.

---

# 👨‍💻 Author

### Ojeshwi Gautam

