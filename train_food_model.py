# =============================================================================
# FoodFocus AI - training script (commented version)
#
# WHAT THIS SCRIPT DOES (big picture):
#   1. Loads a dataset of food images (as URLs) + food names + nutrition info.
#   2. Downloads each image and converts it into numbers a model can read.
#   3. Builds ONE model with TWO jobs:
#        a) Classification  -> "What food is this?"        (e.g. pizza, dosa)
#        b) Regression      -> "How many calories/protein/fat/carbs/fiber?"
#   4. Trains the model, then uploads it to the Hugging Face Hub.
# =============================================================================


# ----------------------------- IMPORTS ---------------------------------------

# load_dataset: downloads datasets from the Hugging Face Hub
# Dataset: the dataset object type (imported but not used directly below)
from datasets import load_dataset, Dataset

# AutoFeatureExtractor: prepares raw images (resize, normalize) for the model
# AutoModelForImageClassification: loads a pretrained image-classification model
# TrainingArguments: holds all training settings (epochs, batch size, etc.)
# Trainer: Hugging Face's ready-made training loop
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, TrainingArguments, Trainer

import torch                 # main deep-learning library (tensors, GPU support)
import torch.nn as nn        # building blocks for neural networks (layers, losses)

# These torchvision transforms are imported but NOT used in this script
# (the feature extractor below already does the resizing/normalizing).
from torchvision.transforms import Compose, RandomResizedCrop, Normalize

from PIL import Image        # Pillow: opens and converts image files
import requests              # makes HTTP requests (used to download images)
from io import BytesIO       # lets us treat downloaded bytes like a file in memory

# train_test_split from sklearn is imported but NOT used
# (the dataset's own .train_test_split() method is used instead).
from sklearn.model_selection import train_test_split
import numpy as np           # numerical library (imported but not used here)

from huggingface_hub import login   # lets this script authenticate with Hugging Face


# ----------------------- HUGGING FACE LOGIN ----------------------------------

# Logs in so the script can push the trained model to your Hugging Face account.
# SECURITY: never paste a real token into code. Put it in an environment
# variable (e.g. os.environ["HF_TOKEN"]) or run `huggingface-cli login` once.
login('YOUR_HF_TOKEN_HERE')


# ----------------------- LOAD THE DATASET ------------------------------------

print("Loading dataset...")
# Downloads the "Foods_data" dataset. Each row has (at least):
#   - imgurl     : link to a food image
#   - name       : the food's name (this becomes the classification label)
#   - nutritions : a string that looks like a Python dict, e.g.
#                  "{'Calories': '250 kcal', 'Protein': '10 g', ...}"
dataset = load_dataset("JeevakrishnaVetrivel/Foods_data")
print("Dataset loaded successfully!")


# ----------------------- IMAGE DOWNLOAD HELPER -------------------------------

def download_image(url):
    """Download an image from a URL and return it as an RGB PIL image.
    Returns None if anything goes wrong (bad link, timeout, not an image...),
    so one broken link doesn't crash the whole run."""
    try:
        response = requests.get(url, timeout=10)           # wait max 10 seconds
        # Read the downloaded bytes as an image; force 3 colour channels (RGB)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception as e:
        print(f"Failed to download image from {url}: {str(e)}")
        return None


# ----------------------- FEATURE EXTRACTOR -----------------------------------

print("Initializing feature extractor...")
# The feature extractor knows exactly how the ViT model expects images:
# resize to 224x224, scale pixel values, normalize with the right mean/std.
# "in21k" = this ViT was pretrained on ImageNet-21k (~14M images).
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    use_fast=True            # use the faster implementation
)


# ----------------------- LABEL MAPPINGS --------------------------------------

# Models work with numbers, not text. So we give every unique food name an ID.
# set(...)  -> removes duplicates
# list(...) -> turns it into an ordered list
all_names = list(set(dataset["train"]["name"]))

# label2id: {"pizza": 0, "dosa": 1, ...}   (name -> number, used for training)
label2id = {label: i for i, label in enumerate(all_names)}

# id2label: {0: "pizza", 1: "dosa", ...}   (number -> name, used for predictions)
id2label = {i: label for label, i in label2id.items()}


# ----------------------- PREPROCESSING FUNCTION ------------------------------

def preprocess_images(examples):
    """Called by dataset.map() on a BATCH of rows (16 at a time).
    For each row it: downloads the image, converts the food name to a label ID,
    and parses the nutrition text into 5 numbers.
    Returns the data in the format the model needs."""
    print(f"Processing batch of {len(examples['imgurl'])} examples...")

    images = []              # successfully downloaded images
    valid_indices = []       # positions of rows that worked (not used later)
    labels = []              # numeric class IDs
    nutritional_values = []  # [calories, protein, fat, carbs, fiber] per image

    # Loop over the 3 columns in parallel: image URL, food name, nutrition text
    for idx, (url, name, nutrition_str) in enumerate(zip(examples["imgurl"], examples["name"], examples["nutritions"])):
        img = download_image(url)
        if img is not None:                      # skip rows whose image failed
            images.append(img)
            labels.append(label2id[name])        # "pizza" -> 0

            # Turn the text "{'Calories': '250 kcal', ...}" into a real dict.
            # NOTE: eval() runs any code in the string - risky on untrusted
            # data. ast.literal_eval(nutrition_str) is the safe replacement.
            nutrition_dict = eval(nutrition_str)

            # For each nutrient: take the text (e.g. '250 kcal'), split on
            # spaces -> ['250', 'kcal'], keep the first part '250', make it a
            # float. If the nutrient is missing, fall back to '0 <unit>'.
            nutritional_values.append([
                float(nutrition_dict.get('Calories', '0 kcal').split()[0]),
                float(nutrition_dict.get('Protein', '0 g').split()[0]),
                float(nutrition_dict.get('Fat', '0 g').split()[0]),
                float(nutrition_dict.get('Carbohydrates', '0 g').split()[0]),
                float(nutrition_dict.get('Fiber', '0 g').split()[0])
            ])
            valid_indices.append(idx)

    # If every download in this batch failed, return nothing.
    if not images:
        return {}

    # Convert the list of PIL images into a tensor of shape
    # (batch, 3, 224, 224) that the ViT model can accept.
    inputs = feature_extractor(images=images, return_tensors="pt")

    # These keys become the new columns in the processed dataset.
    return {
        "pixel_values": inputs.pixel_values,
        "labels": labels,
        "nutritional_values": nutritional_values
    }


# ----------------------- SPLIT + PREPROCESS ----------------------------------

print("Preprocessing dataset...")
train_dataset = dataset["train"]

# Split the data: 80% for training, 20% for validation (checking progress).
# seed=42 makes the split identical every time you run the script.
train_val_split = train_dataset.train_test_split(test_size=0.2, seed=42)

# .map() applies preprocess_images to every batch of 16 rows.
# remove_columns drops the original columns (imgurl, name, nutritions) since
# we've replaced them with pixel_values / labels / nutritional_values.
# NOTE: this downloads every image, so it can take a long time.
processed_train = train_val_split["train"].map(
    preprocess_images,
    batched=True,
    batch_size=16,
    remove_columns=train_dataset.column_names
)

processed_val = train_val_split["test"].map(
    preprocess_images,
    batched=True,
    batch_size=16,
    remove_columns=train_dataset.column_names
)

print(f"Number of unique food classes: {len(label2id)}")


# ----------------------- THE CUSTOM MODEL ------------------------------------

class FoodClassificationModel(nn.Module):
    """A Vision Transformer (ViT) with two outputs:
         1) logits         -> scores for each food class
         2) nutrition_pred -> 5 predicted nutrition numbers
    The ViT 'body' is shared, so learning to recognise food also helps
    the model estimate its nutrition (multi-task learning)."""

    def __init__(self, num_labels):
        super().__init__()

        # Load the pretrained ViT and attach a fresh classification layer
        # sized for OUR number of food classes.
        # ignore_mismatched_sizes=True lets it replace the original head.
        self.vit = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        # Extra small neural network for predicting nutrition:
        #   768 inputs (ViT's feature size) -> 256 -> ReLU -> 5 outputs
        self.nutrition_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 5)  # calories, protein, fat, carbs, fiber
        )

    def forward(self, pixel_values, labels=None, nutritional_values=None):
        """Runs one forward pass: image in -> predictions (and loss) out."""

        # Run the image through ViT; passing labels makes it compute the
        # classification loss for us.
        outputs = self.vit(pixel_values, labels=labels)

        # Grab the image features from the last layer and pool them into one
        # 768-number summary per image.
        # NOTE: this line is likely to FAIL as written - hidden_states is only
        # returned if you call self.vit(..., output_hidden_states=True), and
        # ViTForImageClassification's inner model usually has no pooler.
        # A common fix: use outputs.hidden_states[-1][:, 0] (the [CLS] token).
        hidden_states = self.vit.vit.pooler(outputs.hidden_states[-1])

        # Feed those features into our extra head -> 5 nutrition predictions.
        nutrition_pred = self.nutrition_head(hidden_states)

        # TRAINING MODE: we know the right answers, so compute the loss.
        if labels is not None and nutritional_values is not None:
            # How wrong was the food classification? (cross-entropy)
            loss_cls = outputs.loss

            # How far off were the nutrition numbers? (mean squared error)
            loss_nutrition = nn.MSELoss()(nutrition_pred, nutritional_values)

            # Combine both: classification counts fully, nutrition counts half.
            # The 0.5 is a tunable weight. Calories are much larger numbers
            # than protein/fiber, so you may want to normalize them.
            total_loss = loss_cls + 0.5 * loss_nutrition

            return {"loss": total_loss, "logits": outputs.logits, "nutrition_pred": nutrition_pred}

        # INFERENCE MODE: no answers available, just return predictions.
        return {"logits": outputs.logits, "nutrition_pred": nutrition_pred}


print("Initializing model...")
model = FoodClassificationModel(num_labels=len(label2id))


# ----------------------- TRAINING SETTINGS -----------------------------------

print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",            # where checkpoints are saved locally
    num_train_epochs=10,               # go through the training data 10 times
    per_device_train_batch_size=16,    # images per training step
    per_device_eval_batch_size=16,     # images per validation step
    warmup_steps=500,                  # learning rate ramps up slowly for the first 500 steps
    weight_decay=0.01,                 # small penalty on big weights to reduce overfitting
    logging_dir="./logs",              # TensorBoard logs
    logging_steps=10,                  # log the loss every 10 steps
    eval_steps=100,                    # run validation every 100 steps
    save_steps=100,                    # save a checkpoint every 100 steps
    evaluation_strategy="steps",       # evaluate by steps (newer versions call this eval_strategy)
    save_strategy="steps",             # save by steps (must match evaluation_strategy)
    load_best_model_at_end=True,       # after training, reload the best checkpoint
    push_to_hub=True,                  # upload the model to Hugging Face
    learning_rate=2e-5,                # how big each weight update is (small = safe for fine-tuning)
)


# ----------------------- CUSTOM TRAINER --------------------------------------

class NutritionTrainer(Trainer):
    """The default Trainer only knows about a single loss from the model.
    We override compute_loss so it also passes the nutrition targets in."""

    def compute_loss(self, model, inputs, return_outputs=False):
        # Call our model with the images, class labels, and nutrition targets.
        # The nutrition list is converted to a float tensor and moved to the
        # same device (CPU/GPU) as the images.
        outputs = model(
            pixel_values=inputs["pixel_values"],
            labels=inputs["labels"],
            nutritional_values=torch.tensor(inputs["nutritional_values"], dtype=torch.float32).to(inputs["pixel_values"].device)
        )

        # Pull out the combined loss from the model's output.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Trainer sometimes also wants the raw outputs (e.g. during evaluation).
        return (loss, outputs) if return_outputs else loss


# ----------------------- TRAIN + UPLOAD --------------------------------------

print("Initializing trainer...")
trainer = NutritionTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_train,     # data used to learn
    eval_dataset=processed_val,        # data used to check progress
)

print("Starting training...")
trainer.train()                        # runs the full training loop

print("Pushing model to hub...")
trainer.push_to_hub("food-focus-ai")   # uploads the final model to your HF account

print("Training completed successfully!")v
