# Next-Word Prediction API

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10-blue?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=flat-square&logo=docker)](https://www.docker.com/)
[![AWS ECS](https://img.shields.io/badge/AWS%20ECS-Fargate-ff9900?style=flat-square&logo=amazon-aws)](https://aws.amazon.com/ecs/)

A production-ready FastAPI web service that predicts the next word in a text sequence using an LSTM language model trained on the WikiText-2 dataset. Containerized with Docker and architected for scalable, serverless deployment on AWS ECS Fargate.

---

## The "Why" (Real-World Value)

Next-word prediction is a core building block for modern typing assistants, autocomplete systems, and localized text-generative interfaces. This repository demonstrates how to take a deep learning model (LSTM), wrap it in a lightweight web API, containerize the environment for reproducibility, and deploy it to cloud infrastructure (AWS Fargate) capable of scaling automatically. By deploying as a serverless container, it eliminates the overhead of managing underlying virtual machines while providing high-availability access to down-stream applications.

---

## Tech Stack

*   **Core Frameworks**: Python, FastAPI (Web API), TensorFlow / Keras (Model Training & Inference).
*   **NLP Tools**: NLTK (Word Tokenization), Hugging Face `datasets` (WikiText-2 corpus download).
*   **DevOps & Infrastructure**: Docker (Containerization), AWS ECR (Container Registry), AWS ECS Fargate (Serverless Container Orchestration).

---

## Architecture & System Workflow

```text
               +----------------------------------------------------+
               |                Local / Client App                  |
               +-------------------------+--------------------------+
                                         |
                                HTTP GET (JSON query)
                                         v
               +----------------------------------------------------+
               |                AWS Application LB                 |
               +-------------------------+--------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                            AWS ECS Fargate Task                                   |
|                                                                                   |
|   +--------------------------+          +-------------------------------------+   |
|   |    FastAPI Endpoints     |          |             LSTM Model              |   |
|   |                          |          |                                     |   |
|   |  - Receive input text    | tokenise |  - Embedding Layer (dim: 128)       |   |
|   |  - Tokenize using NLTK   +--------->|  - LSTM Layer (128 units)           |   |
|   |  - Pad sequence to length|          |  - Dense Softmax Output (dim: 20K)  |   |
|   |  - Parse output to word  |<---------+  - Predicts next word probabilities |   |
|   +--------------------------+  argmax  +-------------------------------------+   |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## Quickstart Guide

### Prerequisites
*   Python 3.9 or 3.10 installed.
*   Docker installed and running.
*   AWS CLI installed and configured (`aws configure`).

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/naimul214/Word-Prediction-API.git
cd Word-Prediction-API
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Model Training & Pipeline (Optional)
If you already have `vocab.json` and `next_word_model.keras`, you can skip to Step 3. Otherwise, run the data pipeline and training script:
```bash
# 1. Download WikiText-2, tokenize text, and generate vocabulary
python data_preparation.py

# 2. Train LSTM model for 50 epochs (capped iterations for demonstration)
python train.py
```

### 3. Run FastAPI Locally
Start the local development server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
*   **Interactive API documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
*   **Sample Endpoint Request**:
    ```bash
    curl "http://localhost:8000/predict_next_word?input_text=I%20am"
    ```
    Response:
    ```json
    {
      "predicted_word": "going"
    }
    ```

### 4. Containerize & Run with Docker
Build and verify the container locally:
```bash
# Build the Docker image
docker build -t next-word-api .

# Run the containerized service
docker run -p 8000:8000 next-word-api
```

### 5. Deploy to AWS ECS Fargate
```bash
# 1. Authenticate Docker with AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# 2. Create ECR repository (if not already existing)
aws ecr create-repository --repository-name next-word-api --region us-east-1

# 3. Tag and push image to AWS ECR
docker tag next-word-api:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/next-word-api:latest
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/next-word-api:latest

# 4. Register the ECS task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json --region us-east-1

# 5. Create the ECS service (substitute with your subnet and security group IDs)
aws ecs create-service \
    --cluster next-word-cluster \
    --service-name next-word-service \
    --task-definition next-word-task \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxxx,subnet-yyyyyy],securityGroups=[sg-zzzzzz],assignPublicIp=ENABLED}" \
    --region us-east-1
```

---

## Results & Performance

> [!NOTE]
> *Developer Note: Replace this section with actual test results or performance metrics.*
> *   **Model Performance**: Top-1 Accuracy: `XX%`, Perplexity: `XX`.
> *   **Latency**: Average API response time: `XX ms` under load testing.
> *   **Inference Hardware**: Evaluated on CPU (AWS Fargate 0.25 vCPU / 0.5 GB RAM) at average `XX` requests per second.

---

## Limitations & Future Work

*   **Fixed Sequence Context**: The model uses a fixed sequence window (`maxlen=10`) and padding, which prevents it from utilizing long-term context beyond 10 preceding words.
*   **Model Architecture Limitations**: Standard LSTM layers process text sequentially and can suffer from vanishing gradients over long sequences. Upgrading to a Transformer-based decoder architecture (e.g., GPT-style attention block) would yield more coherent, long-context predictions.
*   **Dynamic Vocabulary Constraints**: The vocabulary size is capped at 20,000 unique words. Words outside this vocabulary default to the `<unk>` token, leading to prediction dropouts.
*   **Inference Latency & Cold Starts**: Loading the full TensorFlow framework in a Fargate container takes ~5-10 seconds. Compiling the model to ONNX Runtime format would reduce container memory footprint, speed up startup times, and decrease CPU inference latency.

---

## Connect

*   **LinkedIn**: [naimul214](https://linkedin.com/in/naimul214)
*   **GitHub**: [naimul214](https://github.com/naimul214)
