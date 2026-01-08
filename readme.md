
# OSUM-EChat: Enhancing End-to-End Empathetic Spoken Chatbot via Understanding-Driven Spoken Dialogue

This repository contains the official supplementary materials (code and demonstrations) for our ICME submission. **OSUM-EChat** is an open-source, end-to-end spoken dialogue system designed to bridge the gap in empathetic machine-human interaction, especially in resource-limited settings.

---

## 📖 Abstract
Empathy is crucial for natural interactions in spoken dialogue systems. **OSUM-EChat** addresses existing challenges—such as over-reliance on massive datasets and insufficient paralinguistic cue extraction—through two key innovations:
1.  **Understanding-Driven Training Strategy**: A three-stage approach that extends large speech understanding models to complex dialogue tasks.
2.  **Linguistic–Paralinguistic Dual Thinking**: A mechanism that integrates paralinguistic understanding (age, gender, emotion and sound event) via a **Chain-of-Thought (CoT)** to generate empathetic responses.

We also introduce the **EChat-1500H** dataset and the **EChat-eval** benchmark to facilitate future research in empathetic speech-to-speech dialogue.

---

## 📁 Repository Structure

As shown in the project snapshot, the materials are organized as follows:

```text
.
├── 📂 OSUM-EChat_code        # Implementation of training and dual-thinking mechanism
└── 📂 OSUM-EChat_demo_page   # Comprehensive video demonstrations of system capabilities

```

---

## 💻 OSUM-EChat_code

This directory provides the necessary codebase to verify the experimental results and technical innovations claimed in our paper.

### Highlights:

* **Dual Thinking Mechanism**: Source code for the Chain-of-Thought pipeline that integrates linguistic content with paralinguistic cues.
* **Three-Stage Training Pipeline**: Scripts implementing the transition from speech understanding to empathetic dialogue generation.
* **Inference & Deployment**: A lightweight setup to test the end-to-end (Speech-to-Speech) latency and response quality.
* **Evaluation (EChat-eval)**: Scripts to run our proposed evaluation framework for measuring empathetic responsiveness.

> **Verification Note**: These scripts demonstrate how our approach reduces reliance on large-scale datasets while maintaining high-quality empathetic interactions.

---

## 🎬 OSUM-EChat_demo_page

The demo videos provide intuitive evidence of OSUM-EChat's performance in real-world scenarios.

### Video Categories:

* **Empathetic Interaction**: Showcasing how the system recognizes paralinguistic cues and adjusts its prosody and response content accordingly.
* **Real-time Spoken Dialogue**: Demonstrating the end-to-end seamless flow of conversation with minimal latency.
* **Cross-Scenario Tests**: Evaluation of the model's robustness across different user profiles (varying ages and emotional states).

---

## 🛡️ Anonymization & Notes

* **Anonymous Submission**: In compliance with ICME double-blind review requirements, all identifying information has been removed from the code, comments, and video metadata.
* **Self-Awareness Feature**: Although OSUM-EChat possesses a **self-awareness capability** , this specific feature has been omitted from the demos to ensure complete anonymity during the review process.
* **Data & Model Weights**: To maintain strict anonymity and comply with file size constraints, the **EChat-1500H dataset** and the **trained model checkpoints (ckpt)** are currently withheld. These assets will be made publicly available in the final open-source release upon the formal acceptance of the paper. 
* **Structural Verification**: The provided code structure includes placeholders and loading logic to demonstrate how the dataset and weights are integrated into the pipeline.
---

## 🚀 Quick Start

### Code & Training
1. Navigate to `OSUM-EChat_code`.
2. Install dependencies via `pip install -r requirements.txt`.
3. Run `python infer_gradio.py` to launch the interactive web interface.
4. Run `bash train.sh` to start the three-stage training pipeline.

### Demo Videos
1. Navigate to `OSUM-EChat_demo_page`.
2. Open `index.html` with any web browser to view the categorized demonstration gallery.
---

*We appreciate the reviewers' time and valuable feedback on our contribution to empathetic AI.*


