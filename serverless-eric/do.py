from google.cloud import aiplatform

aiplatform.init(project="erics-first-project-450223", location="us-central1")

job = aiplatform.CustomTrainingJob(
    display_name="cifar100-resnet18-training",
    script_path="trainer/task.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest",
    requirements=["gcsfs"],
    staging_bucket="gs://your-bucket"
)

job.run(
    machine_type="a2-highgpu-1g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=1,
    args=[
        "--epochs=50",
        "--batch-size=64", 
        "--gcs-bucket=your-bucket"
    ]
)