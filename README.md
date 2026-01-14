# ML Encrypted Traffic Classifier

> **Course project (Cyber Attack Detection Methods):** Two machine learning models for inference over encrypted network traffic 
> (1) **Application classification** and (2) **Traffic attribution classification** — designed for a balanced dataset with limited samples.  
> The solution is delivered as a **Dockerized inference system** that outputs predictions for both tasks.

🔗 ** Demo:** **https://drive.google.com/file/d/1cCYAM0HiYiIMvFm0WW4_EhMRokpG6SQO/view?usp=sharing**

---
## Dataset

The dataset used in this project is **not publicly available** and cannot be shared due to course and competition restrictions.  
It consists of traffic samples from **128 different applications** and **5 traffic attribution (transfer) types**.

The main challenge of this dataset lies in its **highly balanced class distribution** combined with a **limited number of samples per class**, which makes the learning task non-trivial and requires careful model design to ensure generalization and robust inference.

## Key Features
- **Two ML models**
  - **App classifier**: predicts the target application.
  - **Attribution classifier**: predicts the traffic/transfer type.
- **Inference-focused design** for fast and consistent predictions.
- **Docker-based deployment** for easy and reproducible execution.
- Clean separation between **training** and **inference** (models are not included in this repo).
