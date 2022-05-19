import os
from src.config import models
from src.utils import MODEL_FOLDER, get_model_dir
from src.prettify_output import prettify

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
                file_handle = open(f"{model_dir}/{file_name}", 'r')
                pretty_text = prettify(file_handle)
                file_handle.close()
                with open(f"{model_dir}/pretty-Output", "w") as f:
                    f.write(pretty_text)
