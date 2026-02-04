# AI Text Detection System

This project implements a Hybrid AI Text Detection system that combines deep learning representations with linguistic feature analysis to distinguish between human-written and AI-generated text.

## Features

- **Hybrid Architecture:** Utilizes both transformer-based embeddings and explicit linguistic features (perplexity, burstiness, etc.).
- **Linguistic Analysis:** Extracts stylistic and structural features using `spaCy` and `textstat`.
- **Streamlit Interface:** Interactive web application for real-time text analysis.
- **Robust Training:** Includes logic for checkpointing and resuming training.

## Project Structure

- `src/`: Core source code for models, training, and data processing.
- `data/`: Dataset files (CSV, NPY features).
- `models/`: Trained model checkpoints (`.pt` files).
- `app.py`: Main entry point for the Streamlit application.
- `notebooks/`: Jupyter notebooks for analysis and experiments.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- spaCy
- Streamlit
