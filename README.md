# Next-Word Prediction API

A FastAPI-based web service that predicts the next word in a text sequence using an LSTM model trained on the WikiText-2 dataset. This project is deployed on AWS ECS Fargate for scalable, serverless hosting.

---

## User Definition

- **Who**: Developers, students, or researchers interested in NLP applications.
- **Purpose**: To interact with a pre-trained next-word prediction model via a simple HTTP API, for testing, app integration, or educational exploration.
- **Requirements**: Basic knowledge of HTTP requests (e.g., `curl` or browser), Python for local setup, and AWS CLI/Docker for deployment.

---

## Setup Instructions

### Prerequisites

- **Python 3.9+**: For local development ([download](https://www.python.org/downloads/)).
- **Docker**: For containerization ([install](https://docs.docker.com/get-docker/)).
- **AWS CLI**: Configured with credentials ([install](https://aws.amazon.com/cli/), run `aws configure`).
- **AWS Account**: With IAM permissions for ECR and ECS.
- **Git**: Optional, to clone the repository.

Steps to Set Up, Dockerize, and Deploy the Next-Word Prediction API

1. Install Prerequisites
- Python 3.9+: Download from python.org and install. Check with: python --version
- Docker: Install from docker.com. Check with: docker --version
- AWS CLI: Install from aws.amazon.com/cli. Check with: aws --version. Configure with: aws configure (use your Access Key ID, Secret Access Key, region us-east-1, format json)
- Git (optional): Install from git-scm.com. Check with: git --version

2. Navigate to Project Directory
- cd /d D:\School\Computer Vision\Assignment 3\Word-Prediction-API

3. Install Python Dependencies
- pip install -r requirements.txt
- Note: requirements.txt should have: fastapi, uvicorn, tensorflow, nltk, datasets

4. Prepare the Dataset (optional if you have vocab.json)
- python data_preparation.py
- Output: Creates vocab.json

5. Train the Model (optional if you have next_word_model.keras)
- python train.py
- Output: Creates next_word_model.keras

6. Test Locally with FastAPI
- uvicorn app:app --host 0.0.0.0 --port 8000
- Test with: curl "http://localhost:8000/predict_next_word?input_text=I%20am"
- Visit: http://localhost:8000/docs for API docs

7. Create the Dockerfile
- Save this as Dockerfile:
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt_tab')"
COPY next_word_model.keras .
COPY vocab.json .
COPY app.py .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

8. Build the Docker Image
- docker build -t next-word-api .
- Check with: docker images

9. Test the Docker Image Locally
- docker run -p 8000:8000 next-word-api
- Test with: curl "http://localhost:8000/predict_next_word?input_text=I%20am"

10. Create an ECR Repository
- aws ecr create-repository --repository-name next-word-api --region us-east-1

11. Authenticate Docker to ECR
- aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 730335357572.dkr.ecr.us-east-1.amazonaws.com

12. Tag the Docker Image
- docker tag next-word-api:latest 730335357572.dkr.ecr.us-east-1.amazonaws.com/next-word-api:latest

13. Push the Image to ECR
- docker push 730335357572.dkr.ecr.us-east-1.amazonaws.com/next-word-api:latest

14. Create an ECS Cluster
- aws ecs create-cluster --cluster-name next-word-cluster --region us-east-1

15. Create an Execution Role (do this once in AWS Console)
- Go to IAM > Roles > Create Role
- Choose AWS Service > Elastic Container Service > ECS Task
- Attach AmazonECSTaskExecutionRolePolicy
- Name it: ecsTaskExecutionRole
- Note ARN: arn:aws:iam::730335357572:role/ecsTaskExecutionRole

16. Create the Task Definition
- Save this as task-definition.json:
{
    "family": "next-word-task",
    "networkMode": "awsvpc",
    "containerDefinitions": [
        {
            "name": "next-word-container",
            "image": "730335357572.dkr.ecr.us-east-1.amazonaws.com/next-word-api:latest",
            "essential": true,
            "portMappings": [{ "containerPort": 8000, "hostPort": 8000, "protocol": "tcp" }],
            "memory": 512,
            "cpu": 256
        }
    ],
    "requiresCompatibilities": ["FARGATE"],
    "memory": 512,
    "cpu": 256,
    "executionRoleArn": "arn:aws:iam::730335357572:role/ecsTaskExecutionRole"
}

17. Register the Task Definition
- aws ecs register-task-definition --cli-input-json file://task-definition.json --region us-east-1

18. Configure Networking (in AWS Console)
- Subnets: Go to VPC > Subnets, pick 2 public subnets (e.g., subnet-xxx, subnet-yyy)
- Security Group: Go to EC2 > Security Groups > Create
  - Name: next-word-sg
  - Inbound Rule: TCP, Port 8000, Source 0.0.0.0/0
  - Note ID: e.g., sg-xxx

19. Deploy the Service
- aws ecs create-service --cluster next-word-cluster --service-name next-word-service --task-definition next-word-task --desired-count 1 --launch-type FARGATE --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" --region us-east-1
- Replace subnet-xxx, subnet-yyy, sg-xxx with your IDs

20. Get the Public IP
- In AWS Console: ECS > next-word-cluster > next-word-service > Tasks
- Click Task ID, find Public IP (e.g., 54.123.456.78)

21. Test the Deployed API
- curl "http://<public-ip>:8000/predict_next_word?input_text=I%20am"
- Browser: http://<public-ip>:8000/predict_next_word?input_text=I%20am
- Docs: http://<public-ip>:8000/docs
