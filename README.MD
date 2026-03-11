# Legal Document Named Entity Recognition and Summarization

A hybrid approach combining rule-based and deep learning models for Named Entity Recognition (NER) in legal documents, with an additional document summarization feature deployed as a Streamlit web application.

## 🎯 Project Overview

This project implements a sophisticated system for:
- Named Entity Recognition in legal documents using a hybrid approach
- Document summarization using state-of-the-art transformer models
- Web-based interface for easy interaction with the models

## 🏗️ Architecture

The project consists of three main components:

1. **Hybrid NER System**
   - Rule-based NER using legal domain-specific patterns
   - Deep learning models (BiLSTM-CRF, LegalBERT)
   - Hybrid model combining both approaches

2. **Document Summarization**
   - BERT-based extractive summarization
   - T5-based abstractive summarization
   - Model evaluation and comparison

3. **Web Interface**
   - Streamlit-based web application
   - User-friendly document upload and processing
   - Visualization of results

## 📋 Requirements

The project requires Python 3.8+ and the following dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/legal-ner-summarization.git
cd legal-ner-summarization
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
```bash
python scripts/download_models.py
```

## 💻 Usage

### Running the Web Interface

```bash
streamlit run api/streamlit_summarizer.py
```

### Training Models

1. NER Models:
```bash
python models/hybrid/train.py
```

2. Summarization Models:
```bash
python summarization/bertsum_trainer.py
python summarization/t5_trainer.py
```

## 📁 Project Structure

```
├── api/                    # Web interface and API endpoints
├── data/                   # Dataset storage
├── models/                 # NER models
│   ├── bertsum/           # BERT-based summarization
│   ├── bilstm_crf/        # BiLSTM-CRF model
│   ├── hybrid/            # Hybrid NER model
│   └── legalbert/         # LegalBERT implementation
├── summarization/         # Summarization models and utilities
├── scripts/               # Utility scripts
├── logs/                  # Training and evaluation logs
└── results/               # Model outputs and evaluations
```


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LegalBERT for pre-trained legal domain embeddings
- Hugging Face Transformers library
- Streamlit for web interface
- All contributors and maintainers

## 📧 Contact

For any questions or suggestions, please contact [kamaleswars1252k5@gmail.com]
