"""
Prediction script for Raw vs Cooked Food Recognition
Use this script to classify new food images
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class FoodPredictor:
    def __init__(self, model_path='food_classifier_model.h5'):
        """Load the trained model"""
        self.model = keras.models.load_model(model_path)
        self.img_height = 224
        self.img_width = 224
        self.class_names = ['Cooked', 'Raw']  # 0: Cooked, 1: Raw
        
    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((self.img_width, self.img_height))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    
    def predict(self, image_path):
        """Predict whether food is raw or cooked"""
        img_array, original_img = self.preprocess_image(image_path)
        
        # Get prediction
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        
        # Interpret results
        if prediction > 0.5:
            label = 'Raw'
            confidence = prediction * 100
        else:
            label = 'Cooked'
            confidence = (1 - prediction) * 100
        
        return {
            'label': label,
            'confidence': confidence,
            'raw_score': prediction * 100,
            'cooked_score': (1 - prediction) * 100,
            'image': original_img
        }
    
    def predict_batch(self, image_paths):
        """Predict multiple images"""
        results = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                result = self.predict(img_path)
                result['path'] = img_path
                results.append(result)
            else:
                print(f"Warning: {img_path} not found")
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction with confidence scores"""
        result = self.predict(image_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display image
        ax1.imshow(result['image'])
        ax1.axis('off')
        ax1.set_title(f"Prediction: {result['label']}\nConfidence: {result['confidence']:.2f}%", 
                     fontsize=14, fontweight='bold')
        
        # Display confidence bar chart
        categories = ['Cooked', 'Raw']
        scores = [result['cooked_score'], result['raw_score']]
        colors = ['#FF6B6B' if result['label'] == 'Cooked' else '#95E1D3',
                  '#95E1D3' if result['label'] == 'Raw' else '#FF6B6B']
        
        bars = ax2.barh(categories, scores, color=colors)
        ax2.set_xlabel('Confidence (%)', fontsize=12)
        ax2.set_xlim(0, 100)
        ax2.set_title('Classification Scores', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax2.text(score + 2, bar.get_y() + bar.get_height()/2, 
                    f'{score:.1f}%', va='center', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction visualization saved to {save_path}")
        else:
            plt.show()
        
        return result
    
    def batch_visualize(self, image_paths, output_dir='predictions'):
        """Visualize predictions for multiple images"""
        os.makedirs(output_dir, exist_ok=True)
        results = self.predict_batch(image_paths)
        
        # Create a grid visualization
        n_images = len(results)
        cols = 3
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, result in enumerate(results):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            ax.imshow(result['image'])
            ax.axis('off')
            
            color = 'green' if result['confidence'] > 80 else 'orange' if result['confidence'] > 60 else 'red'
            ax.set_title(f"{result['label']} ({result['confidence']:.1f}%)", 
                        fontsize=12, fontweight='bold', color=color)
        
        # Hide empty subplots
        for idx in range(n_images, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/batch_predictions.png', dpi=300, bbox_inches='tight')
        print(f"Batch visualization saved to {output_dir}/batch_predictions.png")
        
        return results


# Example usage
if __name__ == "__main__":
    print("Raw vs Cooked Food Recognition - Prediction")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists('food_classifier_model.h5'):
        print("Error: Model file 'food_classifier_model.h5' not found!")
        print("Please train the model first using train_model.py")
        exit(1)
    
    # Load predictor
    print("Loading model...")
    predictor = FoodPredictor('food_classifier_model.h5')
    print("✓ Model loaded successfully!")
    
    # Example: Predict a single image
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES:")
    print("=" * 60)
    
    print("\n1. Single image prediction:")
    print("   predictor = FoodPredictor('food_classifier_model.h5')")
    print("   result = predictor.predict('path/to/image.jpg')")
    print("   print(result)")
    
    print("\n2. Visualize prediction:")
    print("   predictor.visualize_prediction('path/to/image.jpg')")
    
    print("\n3. Batch prediction:")
    print("   images = ['img1.jpg', 'img2.jpg', 'img3.jpg']")
    print("   results = predictor.predict_batch(images)")
    
    print("\n4. Batch visualization:")
    print("   predictor.batch_visualize(images)")
    
    # Test with example images if they exist
    test_dir = 'test_images'
    if os.path.exists(test_dir):
        test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if test_images:
            print(f"\n\n✓ Found {len(test_images)} test images in '{test_dir}' directory")
            print("Running predictions...")
            
            results = predictor.batch_visualize(test_images)
            
            print("\nResults:")
            print("-" * 60)
            for result in results:
                print(f"{os.path.basename(result['path'])}: {result['label']} ({result['confidence']:.2f}%)")
    else:
        print(f"\n\nTo test predictions, create a '{test_dir}' directory and add food images.")