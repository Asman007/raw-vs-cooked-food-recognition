"""
Dataset Organization Helper Script
Helps organize images into the correct folder structure for training
"""

import os
import shutil
from pathlib import Path
import random

class DatasetOrganizer:
    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir, 'train')
        self.val_dir = os.path.join(base_dir, 'validation')
        self.test_dir = os.path.join(base_dir, 'test')
        
    def create_directory_structure(self):
        """Create the required directory structure"""
        directories = [
            os.path.join(self.train_dir, 'raw'),
            os.path.join(self.train_dir, 'cooked'),
            os.path.join(self.val_dir, 'raw'),
            os.path.join(self.val_dir, 'cooked'),
            os.path.join(self.test_dir, 'raw'),
            os.path.join(self.test_dir, 'cooked')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        print("✓ Directory structure created successfully!")
        print("\nCreated structure:")
        print(f"{self.base_dir}/")
        print("  ├── train/")
        print("  │   ├── raw/")
        print("  │   └── cooked/")
        print("  ├── validation/")
        print("  │   ├── raw/")
        print("  │   └── cooked/")
        print("  └── test/")
        print("      ├── raw/")
        print("      └── cooked/")
        
    def split_data(self, source_raw_dir, source_cooked_dir, 
                   train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        Split images from source directories into train/val/test sets
        
        Args:
            source_raw_dir: Directory containing raw food images
            source_cooked_dir: Directory containing cooked food images
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
        """
        if not (abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01):
            raise ValueError("Ratios must sum to 1.0")
        
        def split_and_copy(source_dir, category):
            """Split images from source directory"""
            if not os.path.exists(source_dir):
                print(f"Warning: {source_dir} does not exist!")
                return
            
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            images = [f for f in os.listdir(source_dir) 
                     if os.path.splitext(f.lower())[1] in image_extensions]
            
            if not images:
                print(f"Warning: No images found in {source_dir}")
                return
            
            # Shuffle images
            random.shuffle(images)
            
            # Calculate split indices
            n_total = len(images)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Copy images to respective directories
            for img in train_images:
                src = os.path.join(source_dir, img)
                dst = os.path.join(self.train_dir, category, img)
                shutil.copy2(src, dst)
            
            for img in val_images:
                src = os.path.join(source_dir, img)
                dst = os.path.join(self.val_dir, category, img)
                shutil.copy2(src, dst)
            
            for img in test_images:
                src = os.path.join(source_dir, img)
                dst = os.path.join(self.test_dir, category, img)
                shutil.copy2(src, dst)
            
            print(f"\n{category.capitalize()} images:")
            print(f"  Total: {n_total}")
            print(f"  Train: {len(train_images)}")
            print(f"  Validation: {len(val_images)}")
            print(f"  Test: {len(test_images)}")
        
        # Create directory structure
        self.create_directory_structure()
        
        print("\nSplitting datasets...")
        split_and_copy(source_raw_dir, 'raw')
        split_and_copy(source_cooked_dir, 'cooked')
        
        print("\n✓ Data splitting completed!")
        
    def verify_dataset(self):
        """Verify the dataset structure and count images"""
        print("\nDataset Verification:")
        print("=" * 60)
        
        def count_images(directory):
            """Count images in directory"""
            if not os.path.exists(directory):
                return 0
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            return len([f for f in os.listdir(directory) 
                       if os.path.splitext(f.lower())[1] in image_extensions])
        
        total_train = 0
        total_val = 0
        total_test = 0
        
        for split_name, split_dir in [('Train', self.train_dir), 
                                       ('Validation', self.val_dir), 
                                       ('Test', self.test_dir)]:
            print(f"\n{split_name} Set:")
            raw_count = count_images(os.path.join(split_dir, 'raw'))
            cooked_count = count_images(os.path.join(split_dir, 'cooked'))
            total = raw_count + cooked_count
            
            print(f"  Raw: {raw_count}")
            print(f"  Cooked: {cooked_count}")
            print(f"  Total: {total}")
            
            if split_name == 'Train':
                total_train = total
            elif split_name == 'Validation':
                total_val = total
            else:
                total_test = total
        
        print("\n" + "=" * 60)
        print(f"Overall Total: {total_train + total_val + total_test} images")
        print("=" * 60)
        
        if total_train == 0:
            print("\n⚠ Warning: No training data found!")
            print("Please add images to the train directory.")
            
    def download_sample_data_info(self):
        """Provide information on where to get sample data"""
        print("\n" + "=" * 60)
        print("WHERE TO GET FOOD IMAGES:")
        print("=" * 60)
        
        print("\n1. Kaggle Datasets:")
        print("   - Food-101: https://www.kaggle.com/datasets/dansbecker/food-101")
        print("   - Fruit Images: https://www.kaggle.com/datasets/moltean/fruits")
        print("   - Raw vs Processed Food (if available)")
        
        print("\n2. Online Image Collections:")
        print("   - Unsplash (free, high-quality): https://unsplash.com/s/photos/food")
        print("   - Pexels (free): https://www.pexels.com/search/food/")
        print("   - Pixabay (free): https://pixabay.com/images/search/food/")
        
        print("\n3. Create Your Own Dataset:")
        print("   - Take photos of raw ingredients (vegetables, meat, eggs, etc.)")
        print("   - Cook the same ingredients and photograph them")
        print("   - Ensure good lighting and clear focus")
        print("   - Aim for at least 100 images per category")
        
        print("\n4. Web Scraping (with permission):")
        print("   - Use tools like google-images-download")
        print("   - Always respect copyright and terms of service")
        print("   - Verify image quality and relevance")
        
        print("\nRECOMMENDED MINIMUM:")
        print("  - 500-1000 images per category for good results")
        print("  - More diverse images = better model performance")
        print("=" * 60)


def main():
    print("=" * 60)
    print("DATASET ORGANIZATION HELPER")
    print("=" * 60)
    
    organizer = DatasetOrganizer()
    
    print("\nOptions:")
    print("1. Create directory structure only")
    print("2. Split existing images into train/val/test")
    print("3. Verify current dataset")
    print("4. Show data source information")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        organizer.create_directory_structure()
        print("\nNext steps:")
        print("1. Add raw food images to data/train/raw/")
        print("2. Add cooked food images to data/train/cooked/")
        print("3. Do the same for validation and test folders")
        print("4. Run: python train_model.py")
        
    elif choice == '2':
        source_raw = input("Enter path to raw food images directory: ").strip()
        source_cooked = input("Enter path to cooked food images directory: ").strip()
        
        train_ratio = float(input("Train ratio (default 0.7): ").strip() or "0.7")
        val_ratio = float(input("Validation ratio (default 0.2): ").strip() or "0.2")
        test_ratio = float(input("Test ratio (default 0.1): ").strip() or "0.1")
        
        organizer.split_data(source_raw, source_cooked, train_ratio, val_ratio, test_ratio)
        
    elif choice == '3':
        organizer.verify_dataset()
        
    elif choice == '4':
        organizer.download_sample_data_info()
        
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()