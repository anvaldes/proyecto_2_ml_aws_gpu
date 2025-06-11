import sagemaker
from sagemaker.estimator import Estimator
import boto3

# ------------------------------------------------------
# Parámetros generales
# ------------------------------------------------------
role = "arn:aws:iam::613602870396:role/SageMakerExecutionRole"
bucket = "proyecto-2-ml"
s3_prefix = "2025_06"
region = "us-east-1"
image_uri = "613602870396.dkr.ecr.us-east-1.amazonaws.com/proyecto_2_distributed_gpu:latest"  # Cambia esto

# Sesiones
boto_sess = boto3.Session(region_name=region)
sagemaker_sess = sagemaker.Session(boto_session=boto_sess)

# ------------------------------------------------------
# Estimador de SageMaker con imagen Docker personalizada
# ------------------------------------------------------
estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.p3.8xlarge", # "ml.p3.8xlarge",  # 4 GPUs (MULTI_GPU)
    max_run=6 * 3600,
    base_job_name="proyecto2-distributed",
    sagemaker_session=sagemaker_sess,
    hyperparameters={
        "s3_path": s3_prefix,
        "output_dir": "/opt/ml/model"
    }
)

# ------------------------------------------------------
# Lanzamiento del trabajo
# ------------------------------------------------------
estimator.fit()
