# Setup and Run Instructions

Follow these steps to set up the project on a new machine.

## 1. Prerequisites

- Ensure **Python 3.8+** is installed.
- Ensure **Git** and **Git LFS** are installed.
  - *Windows:* Download from [git-scm.com](https://git-scm.com/) (LFS is usually included).
  - *Linux/Mac:* `sudo apt install git git-lfs` / `brew install git-lfs`.

## 2. Clone the Repository

```bash
git clone <your-repo-url>
cd REU_Final
git lfs pull  # Download large files (models, datasets)
```

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

**Important:** You must download the spaCy English model:

```bash
python -m spacy download en_core_web_md
```

## 4. Running the Application

To start the interactive web interface:

```bash
streamlit run app.py
```

## 5. Training the Model

To retrain or resume training:

```bash
python src/train.py
```

*Note: The training script supports checkpointing. It will look for `data/train_features.npy` to speed up feature extraction.*
