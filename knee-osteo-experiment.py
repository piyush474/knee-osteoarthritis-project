import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report

###############################################################################
# 1. CONFIGURATION
###############################################################################
DATA_DIR = "Knee_Osteoarthritis_Classification"  # Path to your main data folder
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

# Image settings
ORIG_IMAGE_SIZE = (224, 224)
CROP_TOP = 0     # Adjust if needed
CROP_BOTTOM = 0  # Adjust if needed

# Model settings
NUM_CLASSES = 3  # Normal, Osteopenia, Osteoporosis
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 30


###############################################################################
# 2. DATA PREPROCESSING FUNCTION
###############################################################################
def custom_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Custom preprocessing that can:
    1) Crop top/bottom if needed (CROP_TOP, CROP_BOTTOM).
    2) Perform histogram equalization to enhance contrast.
    3) Normalize to [0, 1].
    
    Args:
        image (np.ndarray): Input image in RGB format with shape (224,224,3).
    
    Returns:
        np.ndarray: Preprocessed image.
    """
    # Optional: Crop the image if needed
    if CROP_TOP > 0 or CROP_BOTTOM > 0:
        h, w, c = image.shape
        image = image[CROP_TOP : h - CROP_BOTTOM, :, :]  # Crop top & bottom

    # Convert to YCrCb to apply histogram equalization on the Y (luminance) channel
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    
    # Apply histogram equalization on the grayscale image
    gray_eq = cv2.equalizeHist(gray)
    
    # Convert the equalized grayscale image back to RGB
    rgb_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)
    
    # Normalize to float32 [0,1]
    preprocessed = rgb_eq.astype(np.float32) / 255.0
    
    return preprocessed


###############################################################################
# 3. DATA LOADING & AUGMENTATION
###############################################################################
def create_generators(train_dir: str,
                      val_dir: str,
                      test_dir: str,
                      batch_size: int = 16,
                      target_size: tuple = (224, 224)) -> tuple:
    """
    Creates training, validation, and test data generators with custom preprocessing.

    Args:
        train_dir (str): Path to training data.
        val_dir (str): Path to validation data.
        test_dir (str): Path to test data.
        batch_size (int): Batch size for generators.
        target_size (tuple): Target size for images before custom preprocessing.

    Returns:
        (train_gen, val_gen, test_gen): Keras ImageDataGenerator objects.
    """
    # Use a custom lambda function in the ImageDataGenerator for on-the-fly preprocessing
    train_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',  # For multiple classes
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen


###############################################################################
# 4. MODEL CREATION
###############################################################################
def build_resnet101_model(input_shape: tuple,
                          num_classes: int,
                          learning_rate: float = 1e-4) -> Model:
    """
    Builds and compiles a ResNet101-based model for multi-class classification.
    
    Args:
        input_shape (tuple): Shape of the input, e.g., (224, 224, 3).
        num_classes (int): Number of output classes.
        learning_rate (float): Learning rate for Adam optimizer.
    
    Returns:
        Model: A compiled Keras Model.
    """
    # Input tensor
    inputs = Input(shape=input_shape)

    # Load ResNet101 with ImageNet weights; exclude the top layers
    base_model = ResNet101(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model.trainable = True  # Fine-tune all layers

    # Global average pooling
    x = GlobalAveragePooling2D()(base_model.output)

    # Classification layer
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create and compile
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


###############################################################################
# 5. TRAINING
###############################################################################
def train_model(model: Model,
                train_gen,
                val_gen,
                epochs: int = 30) -> Model:
    """
    Trains the model using the provided training and validation generators.
    
    Args:
        model (Model): Keras Model to train.
        train_gen: Training data generator.
        val_gen: Validation data generator.
        epochs (int): Number of training epochs.
    
    Returns:
        Model: The trained Keras Model (weights restored to the best checkpoint).
    """
    # Callbacks for early stopping and reducing learning rate
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr]
    )

    return model


###############################################################################
# 6. EVALUATION
###############################################################################
def evaluate_model(model: Model, test_gen) -> None:
    """
    Evaluates the trained model on the test data generator and prints accuracy.
    
    Args:
        model (Model): The trained Keras model.
        test_gen: Test data generator.
    """
    loss, accuracy = model.evaluate(test_gen, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%\n")

    # 2) Generate predictions for the classification report
    #    Note: test_gen.classes gives true labels in the same order as the generator yields.
    predictions = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    # Retrieve class labels from the generator
    class_labels = list(test_gen.class_indices.keys())

    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_labels))


###############################################################################
# 7. MAIN EXECUTION
###############################################################################
def main():
    """
    Main function to orchestrate data loading, model creation, training, and evaluation.
    """
    print("=== Creating data generators ===")
    train_gen, val_gen, test_gen = create_generators(
        TRAIN_DIR, VAL_DIR, TEST_DIR,
        batch_size=BATCH_SIZE,
        target_size=ORIG_IMAGE_SIZE
    )

    print("=== Building ResNet101 model ===")
    model = build_resnet101_model(
        input_shape=(ORIG_IMAGE_SIZE[0], ORIG_IMAGE_SIZE[1], 3),
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE
    )

    print("=== Training model ===")
    model = train_model(model, train_gen, val_gen, epochs=EPOCHS)

    print("=== Evaluating model ===")
    evaluate_model(model, test_gen)

    # (Optional) Save the trained model
    model.save("knee_osteoarthritis_resnet101.h5")
    print("Model saved as knee_osteoarthritis_resnet101.h5")


if __name__ == "__main__":
    main()
