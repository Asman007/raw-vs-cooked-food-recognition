"""
Raw vs Cooked Food Recognition Model
This script trains a CNN model to classify food images as raw or cooked
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class FoodClassifier:
    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.history = None
        
    def create_model(self):
        """Create a CNN model for binary classification"""
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def prepare_data_generators(self, train_dir, val_dir, batch_size=32):
        """Create data generators with augmentation for training"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=50):
        """Train the model with callbacks"""
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        checkpoint = keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr, checkpoint]
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("Training history plot saved as 'training_history.png'")
        
    def save_model(self, filepath='food_classifier_model.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")


# Example usage
if __name__ == "__main__":
    print("Raw vs Cooked Food Recognition - Training Script")
    print("=" * 60)
    
    # Initialize classifier
    classifier = FoodClassifier(img_height=224, img_width=224)
    
    # Create model
    print("\nCreating model architecture...")
    model = classifier.create_model()
    print(model.summary())
    
    # Note: You need to organize your dataset in this structure:
    # data/
    #   train/
    #     raw/
    #       image1.jpg
    #       image2.jpg
    #     cooked/
    #       image1.jpg
    #       image2.jpg
    #   validation/
    #     raw/
    #       image1.jpg
    #     cooked/
    #       image1.jpg
    
    print("\n" + "=" * 60)
    print("DATA STRUCTURE REQUIRED:")
    print("=" * 60)
    print("""
    data/
      train/
        raw/          <- Place raw food images here
        cooked/       <- Place cooked food images here
      validation/
        raw/          <- Place validation raw food images here
        cooked/       <- Place validation cooked food images here
    """)
    print("=" * 60)
    
    # Check if data directories exist
    if os.path.exists('data/train') and os.path.exists('data/validation'):
        print("\nData directories found! Preparing to train...")
        
        # Prepare data
        train_gen, val_gen = classifier.prepare_data_generators(
            'data/train',
            'data/validation',
            batch_size=32
        )
        
        print(f"\nTraining samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Class indices: {train_gen.class_indices}")
        
        # Train model
        print("\nStarting training...")
        classifier.train(train_gen, val_gen, epochs=50)
        
        # Plot results
        classifier.plot_training_history()
        
        # Save model
        classifier.save_model('food_classifier_model.h5')
        
        print("\n✓ Training completed successfully!")
    else:
        print("\n⚠ Data directories not found!")
        print("Please create the data structure shown above and add your images.")
        print("\nYou can download food datasets from:")
        print("- Kaggle: https://www.kaggle.com/datasets")
        print("- Google Images (ensure proper licensing)")
        print("- Food-101 dataset (subset for raw/cooked)")