import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os

class Image_encoder(nn.Module):
    def __init__(self, embed_dim, weights_path="Models/Pretrained/resnet50_weights.pth"):
        super().__init__()
        
        resnet = models.resnet50(weights=None)
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            resnet.load_state_dict(state_dict)
            print("Loaded ResNet50 backbone (Offline)")

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.spatial_neck = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Replaces the Dense Head. Projects localized features directly.
        self.local_proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, x):
        x = self.backbone(x)        # [B, 2048, 7, 7]
        x = self.spatial_neck(x)    # [B, 512, 4, 4]
        
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1) # [B, 16, 512]
        
        return self.local_proj(x)   # [B, 16, embed_dim]

    def train(self, mode=True):
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class Text_encoder(nn.Module):
    class ResidualLSTM(nn.Module):
        def __init__(self, dim, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(dim, dim // 2, num_layers=1, 
                                batch_first=True, bidirectional=True)
            self.norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.norm(x + self.dropout(out))

    def __init__(self, vocab_size, word_dim, hidden_dim, embed_dim, depth=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.proj = nn.Linear(word_dim, hidden_dim)
        
        self.layers = nn.ModuleList([self.ResidualLSTM(hidden_dim) for _ in range(depth)])
        
        # Replaces AttentionPooling. Applied to every token independently.
        self.token_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(0.02),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len]
        # Generate padding mask (assuming 0 is the padding token)
        mask = (x != 0) 
        
        x = self.proj(self.embedding(x)) # [Batch, Seq_Len, Hidden]
        
        for layer in self.layers:
            x = layer(x)
        
        out = self.token_proj(x) # [Batch, Seq_Len, embed_dim]
        
        return out, mask
    
    
class Siamese_model(nn.Module):
    def __init__(self, Image_model, Text_model, device):
        super().__init__()
        self.img_enc = Image_model
        self.txt_enc = Text_model
        self.device = device

    def move_to_device(self, device=None):
        target = device if device else self.device
        self.img_enc.to(target)
        self.txt_enc.to(target)

    def train_mode(self):
        self.img_enc.train()
        self.txt_enc.train()
            
    def eval_mode(self):
        self.img_enc.eval()
        self.txt_enc.eval()        

    def forward(self, img, aug_img, pos_cap, neg_cap):
        # Image extraction -> [B, 16, 512]
        v_main = F.normalize(self.img_enc(img), p=2, dim=-1, eps=1e-8)
        v_aug  = F.normalize(self.img_enc(aug_img), p=2, dim=-1, eps=1e-8)

        # Text extraction -> [B, L, 512] and masks -> [B, L]
        t_pos, m_pos = self.txt_enc(pos_cap)
        t_neg, m_neg = self.txt_enc(neg_cap)
        
        t_pos = F.normalize(t_pos, p=2, dim=-1, eps=1e-8)
        t_neg = F.normalize(t_neg, p=2, dim=-1, eps=1e-8)

        return v_main, v_aug, t_pos, t_neg, m_pos, m_neg

    def predict(self, img, caption_text, threshold=0.5):
        """Inference mode: Computes 1-to-1 similarity directly."""
        self.eval_mode()
        
        with torch.no_grad():
            v_img = F.normalize(self.img_enc(img), p=2, dim=-1, eps=1e-8)
            t_text, mask = self.txt_enc(caption_text)
            t_text = F.normalize(t_text, p=2, dim=-1, eps=1e-8)
            
            # 1-to-1 Cross-Attention logic
            sim_matrix = torch.bmm(v_img, t_text.transpose(1, 2))
            word_scores, _ = torch.max(sim_matrix, dim=1)
            
            # Mask out padding tokens
            word_scores = word_scores * mask.float()
            valid_words = torch.sum(mask.float(), dim=1).clamp(min=1.0)
            
            similarity = torch.sum(word_scores, dim=1) / valid_words
            is_match = (similarity > threshold)
            
            return is_match, similarity 
    
    
    
    
    
    
    
    
    