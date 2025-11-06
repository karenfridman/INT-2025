import tensorflow as tf
import os, cv2, glob, random, imghdr, pickle
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import image_dataset_from_directory
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness, RandomSaturation

# Set a global seed
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

gpus = tf.config.experimental.list_physical_devices('GPU')

# Avoid OMM errors by setting GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# =============
# Functions
# =============

# Cell cropping function
def extract_cell_from_nucleus(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detect purple nucleus
    lower_purple = np.array([110, 40, 40])
    upper_purple = np.array([170, 255, 255])
    nucleus_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Clean small specks
    kernel = np.ones((3, 3), np.uint8)
    nucleus_mask = cv2.morphologyEx(nucleus_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Grow outward to approximate full cell
    grow_mask = nucleus_mask.copy()
    for i in range(25):
        grow_mask = cv2.dilate(grow_mask, kernel, iterations=1)
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 30, 100)
        grow_mask[edges > 0] = 0  # stop at boundaries

    # Find largest contour (the cell)
    contours, _ = cv2.findContours(grow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(c)
    cell_crop = img[y:y+h_box, x:x+w_box]
    return cell_crop

# Remove background
def preprocess_cell_keep_cytoplasm(img):
    img = tf.cast(img, tf.uint8)
    img_np = img.numpy()

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Nucleus (purple-blue) range
    lower_nucleus = np.array([110, 40, 40])
    upper_nucleus = np.array([170, 255, 255])
    nucleus_mask = cv2.inRange(hsv, lower_nucleus, upper_nucleus)

    # Cytoplasm (light pink-violet)
    lower_cytoplasm = np.array([140, 15, 90])
    upper_cytoplasm = np.array([179, 120, 255])
    cytoplasm_mask = cv2.inRange(hsv, lower_cytoplasm, upper_cytoplasm)

    # Combine
    combined_mask = cv2.bitwise_or(nucleus_mask, cytoplasm_mask)

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep largest contour (the cell)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        final_mask = np.zeros_like(combined_mask)
        cv2.drawContours(final_mask, [largest], -1, 255, thickness=cv2.FILLED)
    else:
        final_mask = combined_mask

    # Smooth edges
    final_mask = cv2.GaussianBlur(final_mask, (7, 7), 0)

    # Apply mask to image
    result = cv2.bitwise_and(img_np, img_np, mask=final_mask)

    return result

# =============
# Preprocess images
# =============

# Crop and save to new directory (run only once)
data_dir = "/data"  
cropped_dir = "/data/dataset_cropped"

os.makedirs(cropped_dir, exist_ok=True)

for subfolder in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, subfolder)
    if not os.path.isdir(class_dir):
        continue

    save_subdir = os.path.join(cropped_dir, subfolder)
    os.makedirs(save_subdir, exist_ok=True)

    for filename in tqdm(os.listdir(class_dir), desc=f"Processing {subfolder}"):
        img_path = os.path.join(class_dir, filename)
        crop = extract_cell_from_nucleus(img_path)

        if crop is not None:
            new_name = filename.rsplit('.', 1)[0] + "_cropped.jpg"
            save_path = os.path.join(save_subdir, new_name)
            cv2.imwrite(save_path, crop)

print("✅ All cropped images saved successfully.")

# Make subset
# --- Paths ---
data_dir = "/data/dataset_cropped"  # original full dataset path
filtered_dir = "/data/dataset_cropped_filtered"

selected_classes = ['neutrophil', 'eosinophil', 'lymphocyte', 'monocyte']

# --- Create new filtered dataset ---
os.makedirs(filtered_dir, exist_ok=True)

for cls in selected_classes:
    src = os.path.join(data_dir, cls)
    dst = os.path.join(filtered_dir, cls)
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)

print("Filtered dataset created with classes:", os.listdir(filtered_dir))

# Remove background
source_dir = "/data/dataset_cropped_filtered"     # your dataset folder
output_dir = "/data/dataset_cropped_filtered_BG"    # where to save processed images
os.makedirs(output_dir, exist_ok=True)

# --- Loop through dataset ---
for class_dir in os.listdir(source_dir):
    src_class_path = os.path.join(source_dir, class_dir)
    if not os.path.isdir(src_class_path):
        continue

    dst_class_path = os.path.join(output_dir, class_dir)
    os.makedirs(dst_class_path, exist_ok=True)

    for img_file in tqdm(os.listdir(src_class_path), desc=f"Processing {class_dir}"):
        img_path = os.path.join(src_class_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        processed = preprocess_cell_keep_cytoplasm(img)

        # Save with '_processed' suffix
        file_name = Path(img_file).stem + "_processed.jpg"
        save_path = os.path.join(dst_class_path, file_name)
        cv2.imwrite(save_path, processed)

# =============
# Split data
# =============

# === CONFIG ===
data_dir = "/data/dataset_cropped_filtered_BG"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

class_names = ['neutrophil', 'eosinophil', 'lymphocyte', 'monocyte']
num_classes = len(class_names)

# --- Ratios ---
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1


# === DATA AUGMENTATION (train only) ===
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.15),
    RandomZoom(0.1),
    RandomContrast(0.2),
    RandomBrightness(0.2),
    RandomSaturation(0.2)
], name="data_augmentation")


# === Load + Preprocess ===
def load_and_preprocess(path, label, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    if augment:
        img = data_augmentation(img)
    return img, label


# === Step 1: Collect all file paths and labels ===
all_files = {cls: [] for cls in class_names}

for cls in class_names:
    jpgs = glob.glob(os.path.join(data_dir, cls, "*.jpg"))
    pngs = glob.glob(os.path.join(data_dir, cls, "*.png"))
    files = jpgs + pngs
    print(f"{cls}: {len(files)} images")

    # Shuffle to randomize distribution
    random.seed(SEED)
    random.shuffle(files)

    # Split per class
    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    all_files[cls] = {
        'train': files[:n_train],
        'val': files[n_train:n_train + n_val],
        'test': files[n_train + n_val:]
    }


# === Step 2: Create datasets per split ===
def make_dataset(file_dict, augment=False):
    datasets = []
    for i, cls in enumerate(class_names):
        files = file_dict[cls]
        if files: # Only create dataset if there are files
            ds_paths = tf.data.Dataset.from_tensor_slices(files)
            ds = ds_paths.map(lambda p: load_and_preprocess(p, tf.constant(i, dtype=tf.int32), augment),
                              num_parallel_calls=tf.data.AUTOTUNE)
            datasets.append(ds)
    return datasets


train_datasets = make_dataset({cls: all_files[cls]['train'] for cls in class_names}, augment=True)
val_datasets   = make_dataset({cls: all_files[cls]['val'] for cls in class_names}, augment=False)
test_datasets  = make_dataset({cls: all_files[cls]['test'] for cls in class_names}, augment=False)


# === Step 3: Balanced sampling from each split ===
def make_balanced_dataset(datasets, batch_size=BATCH_SIZE, shuffle=True):
    if not datasets: # Handle the case of empty datasets
        return tf.data.Dataset.from_tensor_slices((np.array([]), np.array([]))).batch(batch_size)

    weights = [1.0 / len(datasets)] * len(datasets)
    ds = tf.data.Dataset.sample_from_datasets(datasets, weights=weights, seed=SEED)
    if shuffle:
        ds = ds.shuffle(1000, seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_balanced_dataset(train_datasets, BATCH_SIZE, shuffle=True)
val_ds   = make_balanced_dataset(val_datasets, BATCH_SIZE, shuffle=False)
test_ds  = make_balanced_dataset(test_datasets, BATCH_SIZE, shuffle=False)


# === Step 4: Verify ===
print("\n✅ Dataset ready:")
for images, labels in train_ds.take(1):
    print("Train batch:", images.shape, labels.numpy())

for images, labels in val_ds.take(1):
    print("Val batch:", images.shape, labels.numpy())

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (data_augmentation(preprocess_input(x)), y),
                        num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y),
                    num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# =============
# Model
# =============

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # freeze base initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = Dropout(0.5)(x)
outputs = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
check_point = ModelCheckpoint('/data/best_4class_blood_model_imp.keras', monitor='val_accuracy', save_best_only=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[reduce_lr, early_stop, check_point]
)

# === Save history ===
os.makedirs("/data/history_logs", exist_ok=True)
with open("/data/history_logs/train_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("✅ Training history saved to '/data/history_logs/train_history.pkl'")

# =============
# Evaluate model
# =============

model.evaluate(test_ds)

#Confusion matrix
y_true = np.array([y.numpy() for _, y in test_ds.unbatch()])

# Convert the test dataset to numpy arrays before predicting
test_images = np.concatenate([x.numpy() for x, _ in test_ds], axis=0)
y_pred = np.argmax(model.predict(test_images), axis=1)


ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred),
                       display_labels=class_names).plot(xticks_rotation=45)
plt.show()

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# Training curves
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

