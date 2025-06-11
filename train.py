import os
import argparse
import time
import json
import warnings
import boto3
from sklearn.metrics import classification_report
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import numpy as np

warnings.filterwarnings("ignore")

def validate_hf_dataset(path):
    print(f"🔎 Validando dataset en: {path}")

    required_files = ["dataset_info.json", "state.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(path, file)):
            raise FileNotFoundError(f"❌ Falta {file} en {path}")

    with open(os.path.join(path, "dataset_info.json")) as f:
        content = f.read()
        if not content.strip():
            raise ValueError("❌ dataset_info.json está vacío")
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ dataset_info.json inválido: {e}")

    arrow_files = [f for f in os.listdir(path) if f.endswith(".arrow")]
    if not arrow_files:
        raise FileNotFoundError("❌ No se encontró ningún archivo .arrow")

    print("✅ Validación completada")

def download_from_s3(bucket_name, prefix, destination_dir):
    print(f"🔽 Descargando de s3://{bucket_name}/{prefix} a {destination_dir}")
    os.makedirs(destination_dir, exist_ok=True)

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith('/'):
                continue
            relative_path = os.path.relpath(key, prefix)
            local_path = os.path.join(destination_dir, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket_name, key, local_path)
            print(f"✅ {key} → {local_path}")

def upload_to_s3(local_dir, bucket_name, s3_prefix):
    print(f"📤 Subiendo resultados a s3://{bucket_name}/{s3_prefix}")
    s3 = boto3.client("s3")

    for root, _, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_dir)
            s3_key = f"{s3_prefix}/{relative_path}"
            s3.upload_file(local_file_path, bucket_name, s3_key)
            print(f"✅ Subido: {s3_key}")

def main():

    #----------------------------------------------------------------------

    # Intentar leer desde CLI o variables de entorno
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3_path', type=str)
    parser.add_argument('--output_dir', type=str)
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"⚠️ Argumentos desconocidos ignorados: {unknown}")

    # Fallback desde JSON de SageMaker si no se pasaron por CLI
    if not args.s3_path or not args.output_dir:
        hyperparams_path = "/opt/ml/input/config/hyperparameters.json"
        if os.path.exists(hyperparams_path):
            print("ℹ️ Cargando parámetros desde JSON de SageMaker...")
            with open(hyperparams_path, "r") as f:
                params = json.load(f)
            args.s3_path = args.s3_path or params.get("s3_path")
            args.output_dir = args.output_dir or params.get("output_dir")

    # Validación final
    if not args.s3_path or not args.output_dir:
        raise ValueError("❌ Debes especificar --s3_path y --output_dir como argumentos o en el JSON de SageMaker")

    #----------------------------------------------------------------------

    s3_prefix = args.s3_path
    output_dir = args.output_dir
    local_path = "/tmp/data"
    bucket_name = "proyecto-2-ml"

    print("📥 Descargando datasets desde S3...")
    download_from_s3(bucket_name, f"{s3_prefix}/train", f"{local_path}/train")
    download_from_s3(bucket_name, f"{s3_prefix}/val", f"{local_path}/val")
    download_from_s3(bucket_name, f"{s3_prefix}/test", f"{local_path}/test")

    print("📚 Cargando datasets...")
    validate_hf_dataset(f"{local_path}/train")
    validate_hf_dataset(f"{local_path}/val")
    validate_hf_dataset(f"{local_path}/test")

    tokenized_train = load_from_disk(f"{local_path}/train")
    tokenized_val = load_from_disk(f"{local_path}/val")
    tokenized_test = load_from_disk(f"{local_path}/test")

    print("🔧 Cargando modelo y tokenizer desde ./modelo_base")
    tokenizer = AutoTokenizer.from_pretrained("./modelo_base")
    model = AutoModelForSequenceClassification.from_pretrained("./modelo_base", num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_steps=0,
        logging_steps=25,
        report_to="none",
        logging_strategy="no", # Cambiar por no luego
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    #----------------------------------------------------------------------

    # Evaluacion: Pre

    print("🧠 Evaluacion: Previa")
    pred_train = trainer.predict(tokenized_train)
    pred_val = trainer.predict(tokenized_val)
    pred_test = trainer.predict(tokenized_test)

    y_true_train = tokenized_train["label"]
    y_true_val = tokenized_val["label"]
    y_true_test = tokenized_test["label"]

    y_pred_train = np.argmax(pred_train.predictions, axis=-1)
    y_pred_val = np.argmax(pred_val.predictions, axis=-1)
    y_pred_test = np.argmax(pred_test.predictions, axis=-1)

    print('Reporte de métricas')

    print("📊 Classification Report: Train")
    print(classification_report(y_true_train, y_pred_train))

    print("📊 Classification Report: Val")
    print(classification_report(y_true_val, y_pred_val))

    print("📊 Classification Report: Test")
    print(classification_report(y_true_test, y_pred_test))


    #----------------------------------------------------------------------

    print("🚀 Entrenando modelo...")

    start = time.time()
    trainer.train()
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"🕒 Tiempo entrenamiento: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    #----------------------------------------------------------------------

    # Evaluacion: Post

    print("🧠 Evaluacion: Post")
    pred_train = trainer.predict(tokenized_train)
    pred_val = trainer.predict(tokenized_val)
    pred_test = trainer.predict(tokenized_test)

    y_true_train = tokenized_train["label"]
    y_true_val = tokenized_val["label"]
    y_true_test = tokenized_test["label"]

    y_pred_train = np.argmax(pred_train.predictions, axis=-1)
    y_pred_val = np.argmax(pred_val.predictions, axis=-1)
    y_pred_test = np.argmax(pred_test.predictions, axis=-1)

    print('Reporte de métricas')

    print("📊 Classification Report: Train")
    print(classification_report(y_true_train, y_pred_train))

    print("📊 Classification Report: Val")
    print(classification_report(y_true_val, y_pred_val))

    print("📊 Classification Report: Test")
    print(classification_report(y_true_test, y_pred_test))


    #----------------------------------------------------------------------

    print("💾 Guardando modelo...")
    trainer.save_model(f"{output_dir}/modelo_final")
    tokenizer.save_pretrained(f"{output_dir}/modelo_final")

    print("📤 Subiendo resultados a S3...")
    upload_to_s3(output_dir, bucket_name, f"{s3_prefix}/outputs")

    print('Finalizado')

if __name__ == "__main__":
    main()


