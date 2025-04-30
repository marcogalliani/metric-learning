import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """Implementation of a CNN designed for metric learning
    
    The CNN is composed by a base model (e.g., ResNet, ViT) and a projection head. 
    The model outputs embeddings which can be optimised to learn a similarity metric.
    
    """
    def __init__(self, base_model,
                 embedding_dim: int,
                 projection_dim: int,
                 proj_hidden_dim: int):
        super().__init__()
        self.base_model = base_model
        self.embedding_dim = embedding_dim # Directly use the provided value
        self.projection_dim = projection_dim

        # Freeze the params of the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # --- Projection Head ---
        # The input dimension depends on the 'final_embedding' calculation below.
        # base_embedding = torch.cat([cls_embedding, context_vector], dim=-1)
        # Both cls_embedding and context_vector have dimension 'embedding_dim'.
        # So, the concatenated dimension is embedding_dim * 2.
        input_proj_dim = self.embedding_dim * 2

        # Set the hidden dimension for the projection MLP
        if proj_hidden_dim is None:
            proj_hidden_dim = input_proj_dim # Default to same dimension as input

        # Simple 2-layer MLP projection head
        self.projection_head = nn.Sequential(
            nn.Linear(input_proj_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim), # Often helps stabilization
            nn.ReLU(),
            nn.Dropout(p=0.2), # Regularization
            nn.Linear(proj_hidden_dim, self.projection_dim)
        )
        
    def get_base_embeddings(self, x):
        """Get the embeddings outputted by the base model

        Args:
            x (np.array): Input images

        Returns:
            np.array: Embedded images
        """
        outputs = self.base_model(x)
        hidden_states = outputs.last_hidden_state

        # Extract CLS token
        cls_embedding = hidden_states[:, 0]

        # Extract patch tokens and apply attention pooling
        patch_tokens = hidden_states[:, 1:]
        attn_weights = F.softmax(torch.matmul(cls_embedding.unsqueeze(1), patch_tokens.transpose(1, 2)), dim=-1)
        context_vector = torch.matmul(attn_weights, patch_tokens).squeeze(1)

        # Combine CLS and attention-pooled context
        base_embedding = torch.cat([cls_embedding, context_vector], dim=-1)
        
        return base_embedding
        
    def forward(self, x):
        
        outputs = self.base_model(x)
        hidden_states = outputs.last_hidden_state

        # Extract CLS token
        cls_embedding = hidden_states[:, 0]

        # Extract patch tokens and apply attention pooling
        patch_tokens = hidden_states[:, 1:]
        attn_weights = F.softmax(torch.matmul(cls_embedding.unsqueeze(1), patch_tokens.transpose(1, 2)), dim=-1)
        context_vector = torch.matmul(attn_weights, patch_tokens).squeeze(1)

        # Combine CLS and attention-pooled context
        base_embedding = torch.cat([cls_embedding, context_vector], dim=-1)
        
        projected_embedding = self.projection_head(base_embedding)

        return projected_embedding
    
    
class ViTNetwork(nn.Module):
    """ 
    Adapting ViT for metric learning
    """
    def __init__(self, base_model, n_unfreeze: int = 0):
        super().__init__()
        self.base_model = base_model

        # Freeze the params of the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze last `n` layers
        # Flatten all parameter layers in order
        all_layers = list(self.base_model.named_parameters())

        for name, param in all_layers[-n_unfreeze:]:
            param.requires_grad = True
            print(f"Unfreezing layer: {name}")
        
    def forward(self, x):
        """Get the embeddings outputted by the base model

        Args:
            x (np.array): Input images

        Returns:
            np.array: Embedded images
        """
        outputs = self.base_model(x)
        hidden_states = outputs.last_hidden_state

        # Extract CLS token
        cls_embedding = hidden_states[:, 0]

        # Extract patch tokens and apply attention pooling
        patch_tokens = hidden_states[:, 1:]
        attn_weights = F.softmax(torch.matmul(cls_embedding.unsqueeze(1), patch_tokens.transpose(1, 2)), dim=-1)
        context_vector = torch.matmul(attn_weights, patch_tokens).squeeze(1)

        # Combine CLS and attention-pooled context
        embeddings = torch.cat([cls_embedding, context_vector], dim=-1)
        
        return embeddings
    