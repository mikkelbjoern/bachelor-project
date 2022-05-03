import fastai
import pandas as pd
from fastai import *
from fastai.vision import *
from fastai.vision.all import *
from fastai.data.all import *
import os

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"DEVICE: {DEVICE}")
HOME_PATH = os.path.expanduser('~')

csv_path = HOME_PATH + "/kaggle-data/HAM10000/HAM10000_metadata.csv"
skin_df = pd.read_csv(csv_path)
skin_df.sort_values(by="image_id")

short_to_full_name_dict = {
    "akiec" : "Bowen's disease", # very early form of skin cancer 
    "bcc" : "basal cell carcinoma" , # basal-cell cancer or white skin cancer
    "bkl" : "benign keratosis-like lesions", # non-cancerous skin tumour
    "df" : "dermatofibroma", # non-cancerous rounded bumps 
    "mel" : "melanoma", # black skin cancer
    "nv" : "melanocytic nevi", # mole non-cancerous
    "vasc" : "vascular lesions", # skin condition
}

img_to_class_dict = skin_df.loc[:, ["image_id", "dx"]] # returns only dx and image id column
img_to_class_dict = img_to_class_dict.to_dict('list')  # returns columns as lists in a dict
img_to_class_dict = {img_id : short_to_full_name_dict[disease] for img_id,disease in zip(img_to_class_dict['image_id'], img_to_class_dict['dx']) } # returns a dict mapping image id to disease name
[x for x in img_to_class_dict.items()][:5]

def get_label_from_dict(path):
    return img_to_class_dict[path.stem] # path.stem returns the filename without suffix

skin_db = DataBlock(
    blocks=(ImageBlock, CategoryBlock), # independent variable x is an image | dependent variabley is a category
    item_tfms=[Resize(450), DihedralItem()], # DihedralItem all 4 90 deg roatations and for each: 2 horizonntal flips -> 8 orientations
    batch_tfms=RandomResizedCrop(size=224, min_scale=0.75, max_scale=1.0), # Picks a random scaled crop of an image and resize it to size
    get_items=get_image_files, # returns all images found in a folder and its subfolders
    splitter=RandomSplitter(valid_pct=0.2, seed=42), # splits the the data in train and valid 70/30 
    get_y=get_label_from_dict, # specifies how to get the label of an image
)

img_path = HOME_PATH + "/kaggle-data/HAM10000/HAM10000_images/"

original_file_pth_db = DataBlock(
    blocks=(
        ImageBlock,
        CategoryBlock,
    ),  # independent variable x is an image | dependent variabley is a category
    item_tfms=[
        Resize(450),
        DihedralItem(),
    ],  # DihedralItem all 4 90 deg roatations and for each: 2 horizonntal flips -> 8 orientations
    batch_tfms=RandomResizedCrop(
        size=224, min_scale=0.75, max_scale=1.0
    ),  # Picks a random scaled crop of an image and resize it to size
    get_items=get_image_files,  # returns all images found in a folder and its subfolders
    splitter=RandomSplitter(
        valid_pct=0.2, seed=42
    ),  # splits the the data in train and valid 70/30
    get_y=lambda x: x.stem,  # specifies how to get the label of an image
)

# Save the validation file_names to a csv file
valid_file_names = []
for file_name in original_file_pth_db.datasets(img_path).valid.items:
    valid_file_names.append(file_name)

valid_df = pd.DataFrame(valid_file_names, columns=['image_pth'])
# Save the df to a csv in the same folder as the script
valid_df.to_csv("valid_file_names.csv", index=False)

# skin_db.summary(img_path) # for debugging
dls = skin_db.dataloaders(img_path, device=DEVICE)  # create dataloader using img_path
# skin_db.summary(img_path) # for debugging
dls = skin_db.dataloaders(img_path, device=DEVICE) # create dataloader using img_path

dls.show_batch(max_n=20, nrows=4) # show a batch of images with labels

learn = cnn_learner(dls, resnet18, metrics=accuracy, opt_func=ranger)
learn.fine_tune(epochs=100, freeze_epochs=3, base_lr=0.002, cbs=MixUp(0.5)) # use mixup with callbacks

learn.save("model_resnet18")