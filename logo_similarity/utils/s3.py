import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class S3Service:
    def __init__(self):
        # Prefer Railway vars if provided, otherwise fallback to MinIO
        self.endpoint = os.getenv("RAILWAY_BUCKET_URL") or os.getenv("MINIO_HOST")
        self.access_key = os.getenv("RAILWAY_ACCESS_ID") or os.getenv("MINIO_USERNAME")
        self.secret_key = os.getenv("RAILWAY_ACCESS_KEY") or os.getenv("MINIO_PASSWORD")
        
        self.region = "us-east-1"
        
        if not all([self.endpoint, self.access_key, self.secret_key]):
            logger.warning("S3 credentials not fully configured in environment variables.")

        self.client = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4'),
            region_name=self.region
        )

    def upload_file(self, file_path, bucket, object_name=None, content_type=None):
        """Upload a file to an S3 bucket"""
        if object_name is None:
            object_name = os.path.basename(file_path)

        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type

        try:
            self.client.upload_file(file_path, bucket, object_name, ExtraArgs=extra_args)
            logger.info(f"Successfully uploaded {file_path} to {bucket}/{object_name}")
            return True
        except ClientError as e:
            logger.error(f"Error uploading file: {e}")
            return False

    def download_file(self, bucket, object_name, local_file_path):
        """Download a file from an S3 bucket to a local path"""
        try:
            # Ensure parent directory exists
            Path(local_file_path).parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(bucket, object_name, str(local_file_path))
            logger.info(f"Successfully downloaded {object_name} from {bucket} to {local_file_path}")
            return True
        except ClientError as e:
            logger.error(f"Error downloading file: {e}")
            return False

    def get_presigned_url(self, bucket, object_name, expiration=3600):
        """Generate a presigned URL to share an S3 object"""
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': object_name},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None

    def create_bucket_if_not_exists(self, bucket_name):
        """Create an S3 bucket if it doesn't already exist"""
        try:
            self.client.head_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} already exists.")
        except ClientError:
            # The bucket does not exist or you have no access.
            try:
                self.client.create_bucket(Bucket=bucket_name)
                logger.info(f"Created bucket {bucket_name}")
            except ClientError as e:
                logger.error(f"Error creating bucket: {e}")
                return False
        return True

# Singleton instance
s3_service = S3Service()
