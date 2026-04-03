"""
AWS Deployment Scripts
Integrates: AWS (boto3, SageMaker)
"""

import boto3
import json
import os
from typing import Dict, Optional
import tarfile
import time

class AWSDeployer:
    """Deploy models to AWS SageMaker and Lambda"""
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None
    ):
        """
        Initialize AWS deployer
        
        Args:
            region_name: AWS region
            aws_access_key_id: AWS access key (optional, uses env if not provided)
            aws_secret_access_key: AWS secret key (optional, uses env if not provided)
        """
        self.region = region_name
        
        # Initialize clients
        session_kwargs = {'region_name': region_name}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs['aws_access_key_id'] = aws_access_key_id
            session_kwargs['aws_secret_access_key'] = aws_secret_access_key
        
        self.session = boto3.Session(**session_kwargs)
        self.sagemaker = self.session.client('sagemaker')
        self.s3 = self.session.client('s3')
        self.lambda_client = self.session.client('lambda')
        self.iam = self.session.client('iam')
    
    # ==========================================
    # S3 OPERATIONS
    # ==========================================
    
    def upload_to_s3(
        self,
        local_path: str,
        bucket_name: str,
        s3_key: str
    ) -> str:
        """
        Upload file to S3
        
        Args:
            local_path: Local file path
            bucket_name: S3 bucket name
            s3_key: S3 object key
            
        Returns:
            S3 URI
        """
        try:
            self.s3.upload_file(local_path, bucket_name, s3_key)
            s3_uri = f"s3://{bucket_name}/{s3_key}"
            print(f"✅ Uploaded to {s3_uri}")
            return s3_uri
        except Exception as e:
            print(f"❌ S3 upload failed: {e}")
            raise
    
    def create_bucket(self, bucket_name: str) -> bool:
        """Create S3 bucket"""
        try:
            if self.region == 'us-east-1':
                self.s3.create_bucket(Bucket=bucket_name)
            else:
                self.s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            print(f"✅ Created bucket: {bucket_name}")
            return True
        except self.s3.exceptions.BucketAlreadyOwnedByYou:
            print(f"ℹ️ Bucket {bucket_name} already exists")
            return True
        except Exception as e:
            print(f"❌ Bucket creation failed: {e}")
            return False
    
    # ==========================================
    # SAGEMAKER DEPLOYMENT
    # ==========================================
    
    def deploy_to_sagemaker(
        self,
        model_name: str,
        model_data_s3_uri: str,
        instance_type: str = "ml.t2.medium",
        initial_instance_count: int = 1,
        framework: str = "sklearn"
    ) -> Dict:
        """
        Deploy model to SageMaker endpoint
        
        Args:
            model_name: Name for the model
            model_data_s3_uri: S3 URI of model artifact (tar.gz)
            instance_type: SageMaker instance type
            initial_instance_count: Number of instances
            framework: ML framework ('sklearn', 'pytorch', 'tensorflow')
            
        Returns:
            Deployment info
        """
        try:
            # Get execution role
            role_arn = self._get_or_create_sagemaker_role()
            
            # Determine container image
            if framework == "sklearn":
                image_uri = self._get_sklearn_image_uri()
            elif framework == "pytorch":
                image_uri = self._get_pytorch_image_uri()
            elif framework == "tensorflow":
                image_uri = self._get_tensorflow_image_uri()
            else:
                raise ValueError(f"Unknown framework: {framework}")
            
            # Create model
            self.sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': image_uri,
                    'ModelDataUrl': model_data_s3_uri
                },
                ExecutionRoleArn=role_arn
            )
            
            # Create endpoint configuration
            endpoint_config_name = f"{model_name}-config"
            self.sagemaker.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[{
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InstanceType': instance_type,
                    'InitialInstanceCount': initial_instance_count
                }]
            )
            
            # Create endpoint
            endpoint_name = f"{model_name}-endpoint"
            self.sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            
            print(f"✅ SageMaker endpoint '{endpoint_name}' is being created...")
            print("   This may take 5-10 minutes.")
            
            return {
                'model_name': model_name,
                'endpoint_name': endpoint_name,
                'endpoint_config_name': endpoint_config_name,
                'status': 'Creating'
            }
            
        except Exception as e:
            print(f"❌ SageMaker deployment failed: {e}")
            raise
    
    def _get_sklearn_image_uri(self) -> str:
        """Get SageMaker scikit-learn container image URI"""
        from sagemaker import image_uris
        return image_uris.retrieve(
            framework='sklearn',
            region=self.region,
            version='1.0-1',
            instance_type='ml.t2.medium'
        )
    
    def _get_pytorch_image_uri(self) -> str:
        """Get SageMaker PyTorch container image URI"""
        from sagemaker import image_uris
        return image_uris.retrieve(
            framework='pytorch',
            region=self.region,
            version='1.12',
            py_version='py38',
            instance_type='ml.t2.medium'
        )
    
    def _get_tensorflow_image_uri(self) -> str:
        """Get SageMaker TensorFlow container image URI"""
        from sagemaker import image_uris
        return image_uris.retrieve(
            framework='tensorflow',
            region=self.region,
            version='2.11',
            instance_type='ml.t2.medium'
        )
    
    def _get_or_create_sagemaker_role(self) -> str:
        """Get or create SageMaker execution role"""
        role_name = "MetaAIBuilderSageMakerRole"
        
        try:
            role = self.iam.get_role(RoleName=role_name)
            return role['Role']['Arn']
        except self.iam.exceptions.NoSuchEntityException:
            # Create role
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }
            
            role = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy)
            )
            
            # Attach policies
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
            )
            
            return role['Role']['Arn']
    
    # ==========================================
    # LAMBDA DEPLOYMENT
    # ==========================================
    
    def deploy_to_lambda(
        self,
        function_name: str,
        zip_file_path: str,
        handler: str = "lambda_function.lambda_handler",
        runtime: str = "python3.10",
        memory_size: int = 512,
        timeout: int = 60
    ) -> Dict:
        """
        Deploy function to AWS Lambda
        
        Args:
            function_name: Lambda function name
            zip_file_path: Path to deployment package (zip)
            handler: Handler function
            runtime: Python runtime
            memory_size: Memory in MB
            timeout: Timeout in seconds
            
        Returns:
            Function info
        """
        try:
            # Get or create Lambda execution role
            role_arn = self._get_or_create_lambda_role()
            
            # Read zip file
            with open(zip_file_path, 'rb') as f:
                zip_content = f.read()
            
            # Create or update function
            try:
                response = self.lambda_client.create_function(
                    FunctionName=function_name,
                    Runtime=runtime,
                    Role=role_arn,
                    Handler=handler,
                    Code={'ZipFile': zip_content},
                    MemorySize=memory_size,
                    Timeout=timeout,
                    Environment={
                        'Variables': {
                            'ENVIRONMENT': 'production'
                        }
                    }
                )
                print(f"✅ Lambda function '{function_name}' created")
            except self.lambda_client.exceptions.ResourceConflictException:
                # Update existing function
                response = self.lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=zip_content
                )
                print(f"✅ Lambda function '{function_name}' updated")
            
            return {
                'function_name': function_name,
                'function_arn': response['FunctionArn'],
                'runtime': runtime,
                'status': 'Active'
            }
            
        except Exception as e:
            print(f"❌ Lambda deployment failed: {e}")
            raise
    
    def _get_or_create_lambda_role(self) -> str:
        """Get or create Lambda execution role"""
        role_name = "MetaAIBuilderLambdaRole"
        
        try:
            role = self.iam.get_role(RoleName=role_name)
            return role['Role']['Arn']
        except self.iam.exceptions.NoSuchEntityException:
            # Create role
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }
            
            role = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy)
            )
            
            # Attach policies
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )
            
            time.sleep(10)  # Wait for role to propagate
            
            return role['Role']['Arn']


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def package_model_for_sagemaker(model_path: str, output_path: str = "model.tar.gz"):
    """Package model for SageMaker deployment"""
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(model_path, arcname=os.path.basename(model_path))
    print(f"✅ Model packaged: {output_path}")
    return output_path


# Example usage
if __name__ == "__main__":
    deployer = AWSDeployer(region_name="us-east-1")
    
    # Example: Upload model to S3
    # deployer.create_bucket("meta-ai-builder-models")
    # deployer.upload_to_s3("model.tar.gz", "meta-ai-builder-models", "models/model.tar.gz")
    
    print("AWS Deployer initialized. Use methods to deploy models.")
