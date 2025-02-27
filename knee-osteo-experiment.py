import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# Set logging level to reduce TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------------
# Custom Preprocessing Function (Segmentation + Equalization)
# ---------------------------
def custom_preprocessing(img):
    """
    Preprocess an input image as per the paper:
      - Assume original resolution is 224x224.
      - Crop 60 pixels from the top and bottom to focus on the knee joint (resulting in 224x104).
      - Convert the cropped image to YUV, equalize the Y channel, and convert back to RGB.
      - Normalize pixel values to [0,1].
    """
    # Ensure image is in uint8 format
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    # Crop 60 pixels from top and bottom; original height assumed to be 224.
    h, w = img.shape[:2]
    if h >= 120:
        cropped = img[60:h-60, :]  # New height becomes h - 120 (typically 224-120=104)
    else:
        cropped = img
    
    # Apply histogram equalization on the luminance channel.
    # Convert RGB to YUV.
    img_yuv = cv2.cvtColor(cropped, cv2.COLOR_RGB2YUV)
    # Equalize the histogram of the Y channel.
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # Convert back to RGB.
    equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    # Since the paper uses the cropped resolution (224 x 104), ensure the image remains this size.
    # (If necessary, we can force a resize.)
    processed = cv2.resize(equalized, (224, 104))
    
    # Normalize pixel values to [0,1]
    processed = processed.astype('float32') / 255.0
    return processed

# ---------------------------
# Data Preparation Functions
# ---------------------------
def preprocess_data(df):
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['label'])
    df = df[['image_path', 'category_encoded']]
    # For compatibility with flow_from_dataframe, convert labels to string
    df['category_encoded'] = df['category_encoded'].astype(str)
    return df

def load_and_preprocess_dataset(data_path, categories, folder="train"):
    image_paths, labels = [], []
    for category in categories:
        category_path = os.path.join(data_path, folder, category)
        for image_name in os.listdir(category_path):
            image_paths.append(os.path.join(category_path, image_name))
            labels.append(category)
    return preprocess_data(pd.DataFrame({"image_path": image_paths, "label": labels}))

def balance_dataset(df):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(df[['image_path']], df['category_encoded'])
    return pd.DataFrame({'image_path': X_resampled['image_path'], 'category_encoded': y_resampled.astype(str)})

def plot_label_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x="label", palette="viridis")
    plt.title("Label Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.show()

# ---------------------------
# Data Generators Using Custom Preprocessing
# ---------------------------
def create_generators(train_df, valid_df, test_df, img_size=(104, 224), batch_size=16):
    # Note: We do not use rescale since our custom preprocessing normalizes the images.
    train_data_gen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    valid_data_gen = ImageDataGenerator(preprocessing_function=custom_preprocessing)
    test_data_gen  = ImageDataGenerator(preprocessing_function=custom_preprocessing)
    
    train_gen = train_data_gen.flow_from_dataframe(
        train_df, 
        x_col='image_path', 
        y_col='category_encoded',
        target_size=img_size, 
        class_mode='sparse', 
        batch_size=batch_size, 
        shuffle=True
    )
    valid_gen = valid_data_gen.flow_from_dataframe(
        valid_df, 
        x_col='image_path', 
        y_col='category_encoded',
        target_size=img_size, 
        class_mode='sparse', 
        batch_size=batch_size, 
        shuffle=False
    )
    test_gen = test_data_gen.flow_from_dataframe(
        test_df, 
        x_col='image_path', 
        y_col='category_encoded',
        target_size=img_size, 
        class_mode='sparse', 
        batch_size=batch_size, 
        shuffle=False
    )
    return train_gen, valid_gen, test_gen

# ---------------------------
# Model Definition: Complete ResNet101 Model per the Paper
# ---------------------------
def create_resnet101_model(input_shape, num_classes=3, learning_rate=1e-4):
    """
    Build the ResNet101 model exactly as described in the paper.
    The model uses a ResNet101 backbone (pretrained on ImageNet) with include_top=False.
    A global average pooling layer and a final dense softmax classifier are added.
    """
    inputs = Input(shape=input_shape)
    # Load ResNet101 with ImageNet weights; allow custom input shape (104, 224, 3)
    base_model = ResNet101(weights="imagenet", include_top=False, input_tensor=inputs)
    
    # Option: Freeze initial layers to retain generic features. Adjust the number of unfrozen layers as needed.
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Optionally add Batch Normalization and Dropout for further regularization.
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    # Final classification layer; change num_classes as required.
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# ---------------------------
# Training and Evaluation Functions
# ---------------------------
def train_model(model, train_gen, valid_gen, epochs=100):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    ]
    history = model.fit(train_gen, validation_data=valid_gen, epochs=epochs, callbacks=callbacks, verbose=1)
    return history

def evaluate_model(model, test_gen):
    predictions = np.argmax(model.predict(test_gen), axis=1)
    report = classification_report(test_gen.classes, predictions, target_names=list(test_gen.class_indices.keys()))
    print("Classification Report:\n", report)
    conf_matrix = confusion_matrix(test_gen.classes, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(test_gen.class_indices.keys()),
                yticklabels=list(test_gen.class_indices.keys()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    # Define dataset paths and categories.
    # (Ensure your directory structure matches: data_path/train/<Category>, etc.)
    data_path = "Knee_Osteoarthritis_Classification"
    categories = ["Normal", "Osteopenia", "Osteoporosis"]
    
    # Load datasets (for training, validation, and testing).
    train_df = load_and_preprocess_dataset(data_path, categories, folder="train")
    train_df = balance_dataset(train_df)
    valid_df = load_and_preprocess_dataset(data_path, categories, folder="val")
    test_df  = load_and_preprocess_dataset(data_path, categories, folder="test")
    
    # Optionally, visualize label distribution.
    # plot_label_distribution(train_df)
    
    # Create data generators using our custom preprocessing.
    # Note: Target size is (height, width) = (104, 224) as per the paper.
    train_gen, valid_gen, test_gen = create_generators(train_df, valid_df, test_df, img_size=(104, 224), batch_size=16)
    
    # Build the ResNet101 model with input shape matching the preprocessed images.
    input_shape = (104, 224, 3)
    num_classes = 3  # Adjust as needed for your classification task.
    model = create_resnet101_model(input_shape, num_classes=num_classes, learning_rate=1e-4)
    
    # Train the model.
    history = train_model(model, train_gen, valid_gen, epochs=100)
    
    # Evaluate the model on the test set.
    evaluate_model(model, test_gen)
