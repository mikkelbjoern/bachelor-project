import os
from src.config import models
from src.utils import MODEL_FOLDER, get_model_dir

def symlink_models():
    for key, model_id in models.items():
        # Check if the symlink already exists
        if os.path.exists(f"{MODEL_FOLDER}/{key}"):
            os.remove(f"{MODEL_FOLDER}/{key}")
        
        os.symlink(f"{get_model_dir(model_id)}", f"{MODEL_FOLDER}/{key}")

    # Do symlinking of the Output file
    for key, model_id in models.items():
        model_dir = get_model_dir(model_id)
        # Look for a file starting with "Output"
        for file_name in os.listdir(model_dir):
            if file_name.startswith("Output"):
                # Check if the symlink already exists
                sym_link = f"{MODEL_FOLDER}/{key}/{file_name}"
                if not os.path.exists(sym_link):
                    os.symlink(f"{model_dir}/{file_name}", sym_link)
