import torch
import torch.nn as nn
from transformers import CLIPModel

class MultimodalFakeNewsClassifier(nn.Module):
    """
    A lightweight multimodal baseline for fake news detection.
    Uses CLIP to encode both image and text, then fuses the features 
    using a simple MLP classification head.
    
    RAG evidence can optionally be fused in the future by concatenating
    retrieved text embeddings to the final feature vector before classification.
    """
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", num_classes=2):
        super().__init__()
        # Load pre-trained CLIP model
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # Freeze CLIP parameters to save memory and compute for a laptop-friendly baseline
        for param in self.clip.parameters():
            param.requires_grad = False
            
        embed_dim = self.clip.config.projection_dim
        
        # Simple fusion head: concatenate image and text embeddings, then classify
        # Future RAG augmentation: Change input dim to support (embed_dim * 3) for RAG features
        self.fusion_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, pixel_values, rag_features=None):
        """
        Forward pass for multimodal input.
        """
        # Use the unified CLIP forward path so we always get projected embeddings
        # across Transformers versions.
        clip_outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
        )

        text_features = clip_outputs.text_embeds
        image_features = clip_outputs.image_embeds

        # Some versions already return normalized embeddings, but this keeps the
        # downstream fusion head consistent and safe across environments.
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Combine text and image features
        fused_features = torch.cat((text_features, image_features), dim=1)
        
        # Placeholder for RAG features fusion
        if rag_features is not None:
            # fused_features = torch.cat((fused_features, rag_features), dim=1)
            # Make sure to adjust the fusion_head dense layer input dimension if enabling this.
            pass
        
        # Classification
        logits = self.fusion_head(fused_features)
        return logits
