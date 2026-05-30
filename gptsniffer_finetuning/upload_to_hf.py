from huggingface_hub import HfApi, login

HF_TOKEN = "YOUR_TOKEN"
login(token=HF_TOKEN)

repo_id = "romangeek/hmcorp_python_gptsniffer" 
folder_to_upload = "./baseline"

api = HfApi()

print(f"Creating repo: {repo_id}...")
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

print(f"Beginning upload from {folder_to_upload}...")
api.upload_folder(
    folder_path=folder_to_upload,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Initial commit: baseline model weights"
)

print("Model uploaded. Link: https://huggingface.co/{repo_id}")