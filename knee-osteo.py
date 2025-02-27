import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Dense, Dropout, 
                                     BatchNormalization, GaussianNoise, Reshape, MultiHeadAttention)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.regularizers import l2

# Set logging level to reduce TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess_data(df):
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['label'])
    df = df[['image_path', 'category_encoded']]
    df['category_encoded'] = df['category_encoded'].astype(str)  # Convert to string
    return df

def load_and_preprocess_dataset(data_path, categories, folder="train"):
    image_paths, labels = [], []
    for category in categories:
        category_path = os.path.join(data_path, folder, category)
        for image_name in os.listdir(category_path):
            image_paths.append(os.path.join(category_path, image_name))
            labels.append(category)
    return preprocess_data(pd.DataFrame({"image_path": image_paths, "label": labels}))

def plot_label_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x="label", palette="viridis")
    plt.title("Label Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.show()

def balance_dataset(df):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(df[['image_path']], df['category_encoded'])
    return pd.DataFrame({'image_path': X_resampled['image_path'], 'category_encoded': y_resampled.astype(str)})

def create_generators(train_df, valid_df, test_df, img_size=(224, 224), batch_size=64):
    train_data_gen = ImageDataGenerator(rescale=1./255, rotation_range=180, width_shift_range=0.2, 
                                    shear_range=0.1, zoom_range=0.2, brightness_range=[0.8, 1.2],
                                    height_shift_range=0.2, horizontal_flip=True)
    valid_data_gen = ImageDataGenerator(rescale=1./255)
    test_data_gen = ImageDataGenerator(rescale=1./255)
    return (
        train_data_gen.flow_from_dataframe(train_df, x_col='image_path', y_col='category_encoded',
                                     target_size=img_size, class_mode='sparse', batch_size=batch_size, shuffle=True),
        valid_data_gen.flow_from_dataframe(valid_df, x_col='image_path', y_col='category_encoded',
                                     target_size=img_size, class_mode='sparse', batch_size=batch_size, shuffle=False),
        test_data_gen.flow_from_dataframe(test_df, x_col='image_path', y_col='category_encoded',
                                     target_size=img_size, class_mode='sparse', batch_size=batch_size, shuffle=False)
    )

def create_xception_model(input_shape, num_classes=3, learning_rate=1e-4):
    inputs = Input(shape=input_shape)
    base_model = Xception(weights="imagenet", input_tensor=inputs, include_top=False)
    base_model.trainable = True  # Unfreeze layers for fine-tuning
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # x = Dense(1024, activation="relu")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # x = Dense(512, activation="relu")(x)
    # x = Dropout(0.5)(x)
    # outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def create_irv2_model(input_shape, num_classes=3, learning_rate=1e-4, pretrained_weights='RadImageNet-IRV2_notop.h5'):
    inputs = Input(shape=input_shape)
    
    # Load InceptionResNetV2 with RadImageNet weights
    base_model = InceptionResNetV2(weights=pretrained_weights, include_top=False, input_tensor=inputs)
    base_model.trainable = True  # Unfreeze layers for fine-tuning
    
    # Adding additional layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    return model

def train_model(model, train_gen, valid_gen, epochs=100, steps_per_epoch=None):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    ]
    return model.fit(train_gen, validation_data=valid_gen, epochs=epochs, callbacks=callbacks, verbose=1)

def evaluate_model(model, test_gen):
    predictions = np.argmax(model.predict(test_gen), axis=1)
    report = classification_report(test_gen.classes, predictions, target_names=list(test_gen.class_indices.keys()))
    print(report)
    conf_matrix = confusion_matrix(test_gen.classes, predictions)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(test_gen.class_indices.keys()),
    #             yticklabels=list(test_gen.class_indices.keys()))
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.show()

if __name__ == "__main__":
    # Load dataset
    data_path = "Knee_Osteoarthritis_Classification"
    categories = ["Normal", "Osteopenia", "Osteoporosis"]
    
    # Split dataset
    train_df = load_and_preprocess_dataset(data_path, categories, "train")
    train_df = balance_dataset(train_df)
    valid_df = load_and_preprocess_dataset(data_path, categories, "val")
    test_df = load_and_preprocess_dataset(data_path, categories, "test")
    
    # Create data generators
    train_gen, valid_gen, test_gen = create_generators(train_df, valid_df, test_df, batch_size=16)

    def generator_wrapper():
        while True:
            for batch in train_gen:
                yield batch

    
    train_dataset = tf.data.Dataset.from_generator(
        generator_wrapper,
        output_types=(tf.float32, tf.int32),
        output_shapes=([None, 224, 224, 3], [None])
    )

    train_dataset = train_dataset.repeat().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Initialize model
    input_shape = (224, 224, 3)
    model = create_xception_model(input_shape)
    # model = create_irv2_model(input_shape, pretrained_weights='RadImageNet-IRV2_notop.h5')
    
    # Train model
    history = train_model(model, train_gen, valid_gen)


    
    # Evaluate model
    evaluate_model(model, test_gen)
