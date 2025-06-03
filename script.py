import kagglehub

# Download latest version
path = kagglehub.dataset_download("cynthiarempel/amazon-us-customer-reviews-dataset")

print("Path to dataset files:", path)