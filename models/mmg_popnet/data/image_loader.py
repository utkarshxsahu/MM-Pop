"""
Image embedding loader for precomputed CLIP embeddings
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch

class ImageEmbeddingLoader:
    """
    Loads precomputed CLIP image embeddings (512-dim, fp16)
    Similar to EmbeddingLoader but simpler since we only need root_post_id lookup
    """
    def __init__(self, image_dir, preload=True):
        """
        Args:
            image_dir: Directory containing clip_embeddings_fp16.npy and image_index.parquet
            preload: Whether to load embeddings into RAM (recommended)
        """
        self.image_dir = image_dir
        
        # Load index mapping root_post_id -> row_idx
        index_path = os.path.join(image_dir, "image_index.parquet")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Image index not found: {index_path}")
        
        print(f"Loading image index from {index_path}...")
        index_df = pd.read_parquet(index_path)
        
        # Create lookup dict: root_post_id -> row_idx
        self.index = index_df.set_index('root_post_id')['row_idx'].to_dict()
        print(f"  Loaded index for {len(self.index)} images")
        
        # Load embeddings
        self.embeddings = None
        if preload:
            embeddings_path = os.path.join(image_dir, "clip_embeddings_fp16.npy")
            if not os.path.exists(embeddings_path):
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
            
            print(f"Loading CLIP embeddings from {embeddings_path}...")
            self.embeddings = np.load(embeddings_path).astype(np.float32)  # Convert to fp32
            print(f"  Shape: {self.embeddings.shape}")
            print(f"  Size: {self.embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    def get_embedding(self, root_post_id):
        """
        Get CLIP embedding for a root post
        
        Args:
            root_post_id: String ID (e.g., 't3_kz2we2')
        
        Returns:
            numpy array (512,) if image exists, None otherwise
        """
        if root_post_id not in self.index:
            return None
        
        row_idx = self.index[root_post_id]
        return self.embeddings[row_idx]
    
    def has_image(self, root_post_id):
        """Check if image exists for root post"""
        return root_post_id in self.index
    
    def clear_cache(self):
        """Free memory"""
        self.embeddings = None


class ImagePixelLoader:
    """
    Loads raw images for unfrozen CLIP training
    """
    def __init__(self, image_dir):
        """
        Args:
            image_dir: Directory containing raw .jpg images
        """
        from transformers import CLIPProcessor  # Use CLIPProcessor to match preprocessing
        
        self.image_dir = image_dir
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Cache available images for fast lookup
        print(f"Indexing images in {image_dir}...")
        if os.path.exists(image_dir):
            self.available_images = set(
                f.replace('.jpg', '') for f in os.listdir(image_dir) 
                if f.endswith('.jpg')
            )
            print(f"  Found {len(self.available_images)} images")
        else:
            self.available_images = set()
            print(f"  Warning: Directory not found")
    
    def get_image(self, root_post_id):
        """
        Load and preprocess image for CLIP
        
        Args:
            root_post_id: String ID (e.g., 't3_kz2we2')
        
        Returns:
            torch.Tensor (3, 224, 224) if image exists, None otherwise
        """
        # Strip t3_ prefix
        clean_id = root_post_id.replace('t3_', '') if root_post_id else None
        
        if not clean_id or clean_id not in self.available_images:
            return None
        
        img_path = os.path.join(self.image_dir, f"{clean_id}.jpg")
        
        try:
            image = Image.open(img_path).convert("RGB")
            # Process returns dict with 'pixel_values' as (1, 3, 224, 224)
            processed = self.processor(images=image, return_tensors="pt")
            return processed['pixel_values'][0]  # Return (3, 224, 224)
        except Exception as e:
            print(f"  Warning: Failed to load {img_path}: {e}")
            return None
    
    def has_image(self, root_post_id):
        """Check if image exists for root post"""
        clean_id = root_post_id.replace('t3_', '') if root_post_id else None
        return clean_id in self.available_images if clean_id else False