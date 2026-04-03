"""
Azure Deployment Scripts
Integrates: Microsoft Azure (Azure ML, Container Instances, Blob Storage)
"""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment, Environment
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from typing import Dict, List

class AzureDeployer:
    """Deploy models to Microsoft Azure"""
    
    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        workspace_name: str,
        location: str = "eastus"
    ):
        """
        Initialize Azure deployer
        
        Args:
            subscription_id: Azure subscription ID
            resource_group: Resource group name
            workspace_name: Azure ML workspace name
            location: Azure region
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.location = location
        
        # Initialize ML client
        credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        print(f"✅ Connected to Azure ML workspace: {workspace_name}")
    
    # ==========================================
    # BLOB STORAGE OPERATIONS
    # ==========================================
    
    def upload_to_blob_storage(
        self,
        local_path: str,
        connection_string: str,
        container_name: str,
        blob_name: str
    ) -> str:
        """
        Upload file to Azure Blob Storage
        
        Args:
            local_path: Local file path
            connection_string: Storage account connection string
            container_name: Container name
            blob_name: Blob name
            
        Returns:
            Blob URL
        """
        try:
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            blob_url = blob_client.url
            print(f"✅ Uploaded to {blob_url}")
            return blob_url
        except Exception as e:
            print(f"❌ Blob upload failed: {e}")
            raise
    
    # ==========================================
    # AZURE ML DEPLOYMENT
    # ==========================================
    
    def deploy_to_azure_ml(
        self,
        model_name: str,
        model_path: str,
        endpoint_name: str,
        instance_type: str = "Standard_DS2_v2",
        instance_count: int = 1,
        environment_name: str = "sklearn-env"
    ) -> Dict:
        """
        Deploy model to Azure ML managed endpoint
        
        Args:
            model_name: Model name
            model_path: Local path to model file
            endpoint_name: Endpoint name
            instance_type: VM instance type
            instance_count: Number of instances
            environment_name: Environment name
            
        Returns:
            Deployment info
        """
        try:
            # Register model
            model = Model(
                path=model_path,
                name=model_name,
                description=f"Meta AI Builder model: {model_name}"
            )
            registered_model = self.ml_client.models.create_or_update(model)
            print(f"✅ Model registered: {registered_model.name} (version {registered_model.version})")
            
            # Create endpoint
            endpoint = ManagedOnlineEndpoint(
                name=endpoint_name,
                description=f"Endpoint for {model_name}",
                auth_mode="key"
            )
            
            try:
                endpoint_result = self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
                print(f"✅ Endpoint created: {endpoint_result.name}")
            except Exception as e:
                print(f"ℹ️ Endpoint may already exist: {e}")
                endpoint_result = self.ml_client.online_endpoints.get(endpoint_name)
            
            # Create deployment
            deployment = ManagedOnlineDeployment(
                name=f"{model_name}-deployment",
                endpoint_name=endpoint_name,
                model=f"{registered_model.name}:{registered_model.version}",
                instance_type=instance_type,
                instance_count=instance_count,
                environment=environment_name
            )
            
            deployment_result = self.ml_client.online_deployments.begin_create_or_update(deployment).result()
            print(f"✅ Deployment created: {deployment_result.name}")
            
            # Set traffic to 100%
            endpoint_result.traffic = {deployment_result.name: 100}
            self.ml_client.online_endpoints.begin_create_or_update(endpoint_result).result()
            
            return {
                'model_name': model_name,
                'model_version': registered_model.version,
                'endpoint_name': endpoint_name,
                'deployment_name': deployment_result.name,
                'scoring_uri': endpoint_result.scoring_uri,
                'status': 'Deployed'
            }
            
        except Exception as e:
            print(f"❌ Azure ML deployment failed: {e}")
            raise
    
    def deploy_sklearn_model(
        self,
        model_name: str,
        model_path: str,
        endpoint_name: str
    ) -> Dict:
        """
        Deploy scikit-learn model to Azure ML
        
        Args:
            model_name: Model name
            model_path: Path to pickled model
            endpoint_name: Endpoint name
            
        Returns:
            Deployment info
        """
        # Create scikit-learn environment
        env = Environment(
            name="sklearn-env",
            description="Scikit-learn environment",
            conda_file="conda.yaml",  # You would need to create this
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
        )
        
        try:
            self.ml_client.environments.create_or_update(env)
        except Exception as e:
            print(f"ℹ️ Environment may already exist: {e}")
        
        return self.deploy_to_azure_ml(
            model_name=model_name,
            model_path=model_path,
            endpoint_name=endpoint_name,
            environment_name="sklearn-env"
        )
    
    # ==========================================
    # PREDICTION
    # ==========================================
    
    def predict(
        self,
        endpoint_name: str,
        instances: List[Dict]
    ) -> List:
        """
        Make predictions using deployed endpoint
        
        Args:
            endpoint_name: Endpoint name
            instances: List of instances to predict
            
        Returns:
            Predictions
        """
        try:
            import requests
            
            # Get endpoint
            endpoint = self.ml_client.online_endpoints.get(endpoint_name)
            
            # Get scoring URI and key
            scoring_uri = endpoint.scoring_uri
            keys = self.ml_client.online_endpoints.get_keys(endpoint_name)
            primary_key = keys.primary_key
            
            # Make request
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {primary_key}'
            }
            
            data = {"data": instances}
            response = requests.post(scoring_uri, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Prediction failed: {response.text}")
                
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            raise
    
    # ==========================================
    # MANAGEMENT
    # ==========================================
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        models = self.ml_client.models.list()
        return [model.name for model in models]
    
    def list_endpoints(self) -> List[str]:
        """List all endpoints"""
        endpoints = self.ml_client.online_endpoints.list()
        return [endpoint.name for endpoint in endpoints]
    
    def delete_endpoint(self, endpoint_name: str):
        """Delete an endpoint"""
        try:
            self.ml_client.online_endpoints.begin_delete(endpoint_name).result()
            print(f"✅ Endpoint deleted: {endpoint_name}")
        except Exception as e:
            print(f"❌ Endpoint deletion failed: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize deployer
    deployer = AzureDeployer(
        subscription_id="your-subscription-id",
        resource_group="your-resource-group",
        workspace_name="your-workspace-name",
        location="eastus"
    )
    
    print("Azure Deployer initialized. Use methods to deploy models.")
