import kagglehub

# Download latest version
path = kagglehub.dataset_download("konradb/ziilogos")

print("Path to dataset files:", path)