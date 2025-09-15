# Differentially Private Synthetic Data Generation

This project demonstrates how **inference-only Large Language Models (LLMs)** can generate **synthetic datasets** from sensitive data, while ensuring **Differential Privacy (DP)**.

🚀 Hosted on **Streamlit Cloud** for free and open public access.

---

## ✨ Features
- Generate realistic **synthetic tabular data**
- Apply **differential privacy** (Laplace noise) during generation
- Download synthetic datasets for safe sharing
- Runs entirely in-browser via **Streamlit**

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) for UI
- [Faker](https://faker.readthedocs.io/) for mock data
- `numpy` for DP noise injection
- Python 3.9+

---

## ▶️ Run Locally
```bash
git clone https://github.com/yourusername/dp-synth-data.git
cd dp-synth-data
pip install -r requirements.txt
streamlit run app.py
