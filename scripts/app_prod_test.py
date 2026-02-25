
import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Production Test App for Logo Similarity
# Connects to the FastAPI backend and retrieves images via Boto3/Minio

st.set_page_config(page_title="Logo Similarity - Prod Test", layout="wide")

# S3 / MinIO Configuration
s3_client = boto3.client(
    's3',
    endpoint_url=os.getenv("MINIO_HOST", "http://192.168.1.98:9000/"),
    aws_access_key_id=os.getenv("MINIO_USERNAME"),
    aws_secret_access_key=os.getenv("MINIO_PASSWORD"),
    config=boto3.session.Config(signature_version='s3v4'),
    region_name='us-east-1'
)
# Use 'l3d' as primary bucket for migrated images
bucket_name = os.getenv("MINIO_BUCKET", "l3d")

def get_image_from_s3(filename):
    """Fetch image bytes directly from S3/MinIO. Handles extension case mismatch."""
    # Try multiple variants if necessary: original, lowercase
    keys_to_try = [f"images/{filename}", f"images/{filename.lower()}"]
    
    # Remove duplicates
    keys_to_try = list(dict.fromkeys(keys_to_try))

    for key in keys_to_try:
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            return response['Body'].read()
        except s3_client.exceptions.NoSuchKey:
            continue
        except Exception as e:
            st.error(f"Error fetching {key}: {e}")
            return None
    
    st.warning(f"Image {filename} not found in bucket {bucket_name} (tried {keys_to_try})")
    return None

st.title("🛡️ Logo Similarity Production Test")
st.markdown("""
This app tests the **Production Backend Pipeline**:
1. Upload an image.
2. The image is sent to the FastAPI container via REST.
3. The backend processes the image (ONNX + PCA) and searches the FAISS index.
4. Matches are retrieved via **Boto3 direct connection** to MinIO (Bucket: `l3d`).
""")

# Configuration
backend_url = st.sidebar.text_input("Backend API URL", value="http://localhost:8000")
top_k = st.sidebar.slider("Top K Results", 1, 100, 20)

# 1. Health Check
try:
    health_resp = requests.get(f"{backend_url}/health", timeout=2)
    if health_resp.status_code == 200:
        health = health_resp.json()
        st.sidebar.success(f"Backend: {health.get('status', 'ready')}")
        st.sidebar.info(f"Index Size: {health.get('index_count', 0)}")
    else:
        st.sidebar.error(f"Backend Error: {health_resp.status_code}")
except Exception as e:
    st.sidebar.error(f"Could not connect to backend: {e}")

# 2. Upload Query
uploaded_file = st.file_uploader("Upload a logo to search...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display query
    col1, col2 = st.columns([1, 4])
    with col1:
        img = Image.open(uploaded_file)
        # st.image(img, caption="Query Logo", width='stretch') # streamlit 1.40+ style
        st.image(img, caption="Query Logo", width='stretch')
        
    with col2:
        if st.button("🔍 Search Production Index"):
            with st.spinner("Processing in container..."):
                try:
                    # Prepare file for multipart upload
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    
                    # Call API
                    response = requests.post(
                        f"{backend_url}/v1/search",
                        params={"top_k": top_k},
                        files=files
                    )
                    
                    if response.status_code == 200:
                        results = response.json().get("results", [])
                        
                        if not results:
                            st.warning("No matches found.")
                        else:
                            st.success(f"Found {len(results)} matches!")
                            
                            # Display grid
                            num_cols = 4
                            for i in range(0, len(results), num_cols):
                                cols = st.columns(num_cols)
                                for j, col in enumerate(cols):
                                    if i + j < len(results):
                                        res = results[i + j]
                                        filename = res['filename']
                                        with col:
                                            # Fetch image bytes using Boto3
                                            img_bytes = get_image_from_s3(filename)
                                            if img_bytes:
                                                st.image(
                                                    img_bytes, 
                                                    caption=f"Score: {res['score']:.4f}\n{filename}",
                                                    width='stretch'
                                                )
                                            else:
                                                st.warning(f"Image {filename} missing")
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

st.divider()
st.caption("Developed for Advanced Agentic Coding - Logo Similarity Project")
