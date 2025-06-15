# 🤖 Distributed Fine-Tuning Pipeline on AWS SageMaker

This repository contains a distributed fine-tuning pipeline using **Hugging Face Transformers**, powered by **PyTorch**, **Accelerate**, and deployed on **AWS SageMaker** using a **custom Docker image with GPU support**.

---

## 🚀 Features

- Distributed training on a single instance with multiple GPUs (`ml.p3.8xlarge`)
- Uses 🤗 `transformers`, `datasets`, `accelerate`, `sklearn`, and `boto3`
- Loads datasets in Hugging Face format from S3
- Pre/post evaluation with `classification_report`
- Uploads trained model and results to S3
- Fully customizable via `train.py` and `config_accelerate.yaml`

---

## 🐳 Docker Image

This project includes a `Dockerfile` based on `pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime`.  
To build and push the image to ECR:

```bash
docker buildx build --platform linux/amd64 -t proyecto_2_distributed_gpu .
docker tag proyecto_2_distributed_gpu 613602870396.dkr.ecr.us-east-1.amazonaws.com/proyecto_2_distributed_gpu:latest
docker push 613602870396.dkr.ecr.us-east-1.amazonaws.com/proyecto_2_distributed_gpu:latest
```

---

## ☁️ Launch Training on SageMaker

To launch the distributed fine-tuning job:

```bash
python launch_distributed_training.py
```

Make sure the image is already in **ECR**, and that your S3 bucket `proyecto-2-ml` contains properly formatted Hugging Face datasets in:

- `s3://proyecto-2-ml/2025_06/train/`
- `s3://proyecto-2-ml/2025_06/val/`
- `s3://proyecto-2-ml/2025_06/test/`

---

## 🔧 Configuration

Accelerate is configured via `config_accelerate.yaml`:

```yaml
distributed_type: MULTI_GPU
num_processes: 4
gpu_ids: all
compute_environment: LOCAL_MACHINE
```

You can modify this file to change precision, use CPU, or set mixed precision modes.

---

## 📁 Output Structure

After training, the pipeline uploads the following to S3:

```
s3://proyecto-2-ml/2025_06/outputs/
│
├── modelo_final/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── ...
```

You can later download this model to continue training, deploy to an endpoint, or use for inference.

---

## 🧪 Local Testing

To test the code locally (on a machine with GPU):

```bash
python train.py --s3_path 2025_06 --output_dir ./outputs
```

---

## 📦 Dependencies

List of key Python packages:

```
pandas==2.0.3
datasets==3.1.0
transformers==4.46.3
torch==2.4.1
accelerate==1.0.1
scikit-learn==1.3.2
boto3==1.34.97
s3fs==2024.3.1
```

---

## 📂 File Overview

- `train.py`: Core training loop, evaluation, and S3 upload logic
- `Dockerfile`: Docker config for SageMaker
- `requirements.txt`: Required Python libraries
- `config_accelerate.yaml`: Config for 🤗 Accelerate
- `launch_distributed_training.py`: Script to launch the job on AWS

---

## ✅ Requirements

- AWS account with SageMaker, ECR, and S3 access
- IAM role with SageMaker permissions (`SageMakerExecutionRole`)
- Proper Hugging Face dataset structure uploaded to S3
- GPU instance availability in your AWS region

---
