import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import random
import numpy as np
import os

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DeblurDataset(Dataset):
    def __init__(self, root_dir, split='train', crop_size=256):
        """
        Dataset for deblurring training/validation with support for multiple image extensions
        
        Args:
            root_dir: Root directory containing 'sharp' and 'blur' subdirectories
            split: 'train' or 'val'
            crop_size: Size of training patches (only used during training)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.crop_size = crop_size
        
        # Get sharp images with multiple extensions
        self.sharp_dir = self.root_dir / 'sharp'
        self.blur_dir = self.root_dir / 'blur'
        
        # Common image extensions
        self.extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        
        # Get all sharp images
        self.sharp_files = []
        for ext in self.extensions:
            self.sharp_files.extend(list(self.sharp_dir.glob(f'*{ext}')))
            self.sharp_files.extend(list(self.sharp_dir.glob(f'*{ext.upper()}')))
        
        # Sort the files to ensure deterministic behavior
        self.sharp_files = sorted(self.sharp_files)
        
        print(f"Found {len(self.sharp_files)} images in {self.sharp_dir}")
        
        # Basic transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Augmentation transforms for training
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
        ]) if split == 'train' else None
        
    def __len__(self):
        return len(self.sharp_files)
    
    def get_random_crop_params(self, img):
        """Get random crop parameters"""
        w, h = img.size
        th, tw = self.crop_size, self.crop_size
        if w == tw and h == th:
            return 0, 0, h, w
        if w < tw or h < th:
            # Handle images smaller than patch size by resizing
            scale = max(tw / w, th / h) * 1.1  # Scale up with a small margin
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BICUBIC)
            w, h = new_w, new_h
        
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw, img
    
    def __getitem__(self, idx):
        try:
            # Load sharp image
            sharp_path = self.sharp_files[idx]
            
            # Get corresponding blur image with same name
            file_name = sharp_path.name
            blur_path = self.blur_dir / file_name
            
            # If blur file doesn't exist with exact name, try matching without extension
            if not blur_path.exists():
                stem = sharp_path.stem
                for ext in self.extensions:
                    candidate = self.blur_dir / f"{stem}{ext}"
                    if candidate.exists():
                        blur_path = candidate
                        break
                    
                    # Also try with uppercase extension
                    candidate = self.blur_dir / f"{stem}{ext.upper()}"
                    if candidate.exists():
                        blur_path = candidate
                        break
            
            # If still no match, use a fallback
            if not blur_path.exists():
                print(f"Warning: No matching blur image for {file_name}")
                # Return a random sample as fallback
                return self.__getitem__(random.randint(0, len(self) - 1))
            
            # Open images with PIL
            try:
                sharp_img = Image.open(sharp_path).convert('RGB')
                blur_img = Image.open(blur_path).convert('RGB')
            except Exception as e:
                print(f"Error loading images: {e}")
                # Return a random sample as fallback
                return self.__getitem__(random.randint(0, len(self) - 1))
            
            # Ensure both images have the same size
            if sharp_img.size != blur_img.size:
                blur_img = blur_img.resize(sharp_img.size, Image.BICUBIC)
            
            # Random crop for training
            if self.split == 'train':
                # Handle random cropping with potential resizing
                i, j, h, w, sharp_img_resized = self.get_random_crop_params(sharp_img)
                if sharp_img_resized is not sharp_img:  # If image was resized
                    sharp_img = sharp_img_resized
                    blur_img = blur_img.resize(sharp_img.size, Image.BICUBIC)
                
                # Crop both images to the same region
                sharp_img = sharp_img.crop((j, i, j + w, i + h))
                blur_img = blur_img.crop((j, i, j + w, i + h))
                
                # Apply augmentation
                if random.random() > 0.5 and self.augment:
                    state = torch.get_rng_state()
                    sharp_img = self.augment(sharp_img)
                    torch.set_rng_state(state)
                    blur_img = self.augment(blur_img)
            
            # Convert to tensors
            sharp_tensor = self.transform(sharp_img)
            blur_tensor = self.transform(blur_img)
            
            return blur_tensor, sharp_tensor
            
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            # Return a random sample as fallback
            return self.__getitem__(random.randint(0, len(self) - 1))


def create_dataloaders(root_dir_train, root_dir_val, batch_size=8, crop_size=256, num_workers=4):
    """Create training and validation dataloaders"""
    train_dataset = DeblurDataset(root_dir_train, split='train', crop_size=crop_size)
    val_dataset   = DeblurDataset(root_dir_val, split='train', crop_size=crop_size)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader