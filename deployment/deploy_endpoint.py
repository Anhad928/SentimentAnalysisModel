from sagemaker.pytorch import PyTorchModel
import sagemaker

def deploy_endpoint():
    sagemaker.Session()
    role = "arn:aws:iam::767397834308:role/sentiment-analysis-deploy-endpoint-role"

    model_uri = "s3://sentiment-analysis-saas-ml/inference/model.tar.gz"
    
    model = PyTorchModel(
        model_data = model_uri,
        role = role,
        framework_version = "2.5.1",
        py_version = "py311",
        entry_point = "inference.py",
        source_dir = ".",
        name = "sentiment-analysis-endpoint"
    )
    
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.xlarge",
        endpoint_name="sentiment-analysis-endpoint"
    )

if __name__ == "__main__":
    deploy_endpoint()