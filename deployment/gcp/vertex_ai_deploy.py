"""
GCP Deployment Scripts
Integrates: Google Cloud Platform (Vertex AI, Cloud Run, Cloud Storage)
"""

from google.cloud import aiplatform, storage
from typing import Dict, Optional, List
import os

class GCPDeployer:
    """Deploy models to Google Cloud Platform"""
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        credentials_path: Optional[str] = None
    ):
        """
        Initialize GCP deployer
        
        Args:
            project_id: GCP project ID
            location: GCP region
            credentials_path: Path to service account JSON (optional)
        """
        self.project_id = project_id
        self.location = location
        
        # Set credentials if provided
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        # Initialize AI Platform
        aiplatform.init(project=project_id, location=location)
        
        # Initialize Storage client
        self.storage_client = storage.Client(project=project_id)
    
    # ==========================================
    # CLOUD STORAGE OPERATIONS
    # ==========================================
    
    def upload_to_gcs(
        self,
        local_path: str,
        bucket_name: str,
        blob_name: str
    ) -> str:
        """
        Upload file to Google Cloud Storage
        
        Args:
            local_path: Local file path
            bucket_name: GCS bucket name
            blob_name: Blob name in bucket
            
        Returns:
            GCS URI
        """
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            
            gcs_uri = f"gs://{bucket_name}/{blob_name}"
            print(f"✅ Uploaded to {gcs_uri}")
            return gcs_uri
        except Exception as e:
            print(f"❌ GCS upload failed: {e}")
            raise
    
    def create_bucket(self, bucket_name: str) -> bool:
        """Create GCS bucket"""
        try:
            self.storage_client.create_bucket(bucket_name, location=self.location)
            print(f"✅ Created bucket: {bucket_name}")
            return True
        except Exception as e:
            print(f"❌ Bucket creation failed: {e}")
            return False
    
    # ==========================================
    # VERTEX AI DEPLOYMENT
    # ==========================================
    
    def deploy_to_vertex_ai(
        self,
        model_name: str,
        model_artifact_uri: str,
        serving_container_image_uri: str,
        machine_type: str = "n1-standard-4",
        min_replica_count: int = 1,
        max_replica_count: int = 3
    ) -> Dict:
        """
        Deploy model to Vertex AI
        
        Args:
            model_name: Name for the model
            model_artifact_uri: GCS URI of model artifacts
            serving_container_image_uri: Container image for serving
            machine_type: Machine type for deployment
            min_replica_count: Minimum replicas
            max_replica_count: Maximum replicas
            
        Returns:
            Deployment info
        """
        try:
            # Upload model
            model = aiplatform.Model.upload(
                display_name=model_name,
                artifact_uri=model_artifact_uri,
                serving_container_image_uri=serving_container_image_uri
            )
            
            print(f"✅ Model uploaded: {model.resource_name}")
            
            # Create endpoint
            endpoint = aiplatform.Endpoint.create(
                display_name=f"{model_name}-endpoint"
            )
            
            print(f"✅ Endpoint created: {endpoint.resource_name}")
            
            # Deploy model to endpoint
            model.deploy(
                endpoint=endpoint,
                deployed_model_display_name=model_name,
                machine_type=machine_type,
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
                traffic_percentage=100
            )
            
            print("✅ Model deployed to endpoint")
            
            return {
                'model_name': model_name,
                'model_resource_name': model.resource_name,
                'endpoint_resource_name': endpoint.resource_name,
                'endpoint_url': endpoint.resource_name,
                'status': 'Deployed'
            }
            
        except Exception as e:
            print(f"❌ Vertex AI deployment failed: {e}")
            raise
    
    def deploy_sklearn_model(
        self,
        model_name: str,
        model_gcs_uri: str,
        sklearn_version: str = "1.0"
    ) -> Dict:
        """
        Deploy scikit-learn model to Vertex AI
        
        Args:
            model_name: Model name
            model_gcs_uri: GCS URI of pickled model
            sklearn_version: Scikit-learn version
            
        Returns:
            Deployment info
        """
        # Use pre-built scikit-learn container
        container_uri = f"us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.{sklearn_version}:latest"
        
        return self.deploy_to_vertex_ai(
            model_name=model_name,
            model_artifact_uri=model_gcs_uri,
            serving_container_image_uri=container_uri
        )
    
    # ==========================================
    # CLOUD RUN DEPLOYMENT
    # ==========================================
    
    def deploy_to_cloud_run(
        self,
        service_name: str,
        image_uri: str,
        region: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        memory: str = "2Gi",
        cpu: str = "2"
    ) -> Dict:
        """
        Deploy container to Cloud Run
        
        Args:
            service_name: Service name
            image_uri: Container image URI (e.g., gcr.io/project/image:tag)
            region: Deployment region
            env_vars: Environment variables
            memory: Memory allocation
            cpu: CPU allocation
            
        Returns:
            Service info
        """
        try:
            from google.cloud import run_v2
            
            region = region or self.location
            client = run_v2.ServicesClient()
            
            # Build service configuration
            service = run_v2.Service()
            service.template.containers = [run_v2.Container(
                image=image_uri,
                resources=run_v2.ResourceRequirements(
                    limits={"memory": memory, "cpu": cpu}
                )
            )]
            
            if env_vars:
                service.template.containers[0].env = [
                    run_v2.EnvVar(name=k, value=v)
                    for k, v in env_vars.items()
                ]
            
            # Deploy service
            parent = f"projects/{self.project_id}/locations/{region}"
            request = run_v2.CreateServiceRequest(
                parent=parent,
                service=service,
                service_id=service_name
            )
            
            operation = client.create_service(request=request)
            response = operation.result()
            
            print(f"✅ Cloud Run service deployed: {response.uri}")
            
            return {
                'service_name': service_name,
                'service_uri': response.uri,
                'region': region,
                'status': 'Active'
            }
            
        except Exception as e:
            print(f"❌ Cloud Run deployment failed: {e}")
            raise
    
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
            endpoint_name: Endpoint resource name
            instances: List of instances to predict
            
        Returns:
            Predictions
        """
        try:
            endpoint = aiplatform.Endpoint(endpoint_name)
            predictions = endpoint.predict(instances=instances)
            return predictions.predictions
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize deployer
    deployer = GCPDeployer(
        project_id="your-project-id",
        location="us-central1"
    )
    
    print("GCP Deployer initialized. Use methods to deploy models.")
