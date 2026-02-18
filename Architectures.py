import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os



class Image_encoder(nn.Module):
    def __init__(self, embed_dim, weights_path="Models/Pretrained/resnet50_weights.pth"):
        super().__init__()
        
        # 1. Init ResNet50
        self.backbone = models.resnet50(weights=None)
        
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state_dict)
            print(f"Loaded ResNet50 backbone (Offline)")

        num_features = self.backbone.fc.in_features 
        self.backbone.fc = nn.Identity()

        # 2. Head
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features // 2), 
            nn.LayerNorm(num_features // 2), 
            nn.LeakyReLU(0.02),
            nn.Dropout(0.1),
        
            nn.Linear(num_features // 2, embed_dim),    
            nn.LayerNorm(embed_dim),          
            nn.LeakyReLU(0.02),
        
            nn.Linear(embed_dim, embed_dim)             
        )
        
    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)

    # --- THE FIX IS HERE ---
    def train(self, mode=True):
        """
        Overwrites the default train() to ensure BN layers in the backbone
        remain in eval mode (frozen stats) even during fine-tuning.
        """
        super().train(mode)
        
        # Force BatchNorm layers in the backbone to stay in Eval mode
        # This prevents the "running_mean" from being corrupted by your batches
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class Text_encoder(nn.Module):
    class ResidualLSTM(nn.Module):
        def __init__(self, dim, dropout=0.1):
            super().__init__()
            # Bi-LSTM: dim // 2 w każdą stronę daje łącznie dim na wyjściu
            self.lstm = nn.LSTM(dim, dim // 2, num_layers=1, 
                                batch_first=True, bidirectional=True)
            self.norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.norm(x + self.dropout(out))

    class AttentionPooling(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.att_weights = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.Tanh(),
                nn.Linear(dim // 2, 1)
            )

        def forward(self, x):
            # x shape: [Batch, Seq_Len, Dim]
            weights = self.att_weights(x) # [Batch, Seq_Len, 1]
            weights = F.softmax(weights, dim=1)
            
            # Ważona suma wektorów słów
            pooled = torch.sum(x * weights, dim=1) # [Batch, Dim]
            return pooled

    def __init__(self, vocab_size, word_dim, hidden_dim, embed_dim, depth=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.proj = nn.Linear(word_dim, hidden_dim)
        
        # Redukujemy głębokość do 3 warstw - dla captioningu to zazwyczaj sweet spot
        self.layers = nn.ModuleList([self.ResidualLSTM(hidden_dim) for _ in range(depth)])
        
        # Nowy moduł agregacji
        self.attention = self.AttentionPooling(hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),    # <--- THE FIX (Safe & Stable)
            nn.LeakyReLU(0.02),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        # x: [Batch, Seq_Len]
        x = self.proj(self.embedding(x)) # [Batch, Seq_Len, Hidden]
        
        for layer in self.layers:
            x = layer(x)
        
        # Zastępujemy torch.max() przez Attention Pooling
        pooled = self.attention(x)
        
        return self.fc(pooled)




class Siamese_model(nn.Module):
    """
    Model which merges 2 models, text and image encoder, and then compares their produced vectors to determine
    if the text describes the image well, and if there are not any mismatches in the caption
    """
    def __init__(self, Image_model, Text_model, device):
        super().__init__()
        self.img_enc = Image_model
        self.txt_enc = Text_model
        
        self.device = device

    def move_to_device(self, device = None):
        if device is None:
            self.img_enc.to(self.device)
            self.txt_enc.to(self.device)
        else:
            self.img_enc.to(device)
            self.txt_enc.to(device)

    def train_mode(self):
        self.img_enc.train()
        self.txt_enc.train()
            
    def eval_mode(self):
        self.img_enc.eval()
        self.txt_enc.eval()       


    def forward(self, img, aug_img, pos_cap, neg_cap):
        """
        For training, we use here the img and augmented image, also the positive and negative caption
        """
        #Visual transformation
        v_main = self.img_enc(img)
        v_aug  = self.img_enc(aug_img)

        #Text transformation
        t_pos = self.txt_enc(pos_cap)
        t_neg = self.txt_enc(neg_cap)

        #NORMALIZATION , specifically for the contrastive and triplet loss
        v_main = F.normalize(v_main, p=2, dim=1, eps=1e-6) # Added eps
        v_aug  = F.normalize(v_aug, p=2, dim=1, eps=1e-6)
        t_pos  = F.normalize(t_pos, p=2, dim=1, eps=1e-6)
        t_neg  = F.normalize(t_neg, p=2, dim=1, eps=1e-6)

        return v_main, v_aug, t_pos, t_neg
    
    
    def predict(self, img, caption_text, threshold=0.5):
        """
        For actual using of the model
        
        It returns if the img and caption are the match as well as the similarity score it produced (0-1)
        """
        
        # Switch to evaluation mode
        self.eval_mode()
        self.eval()
        
        with torch.no_grad():
            #Text and visual embedding from trained submodels / feature extractors 
            v_img = self.img_enc(img)
            t_text = self.txt_enc(caption_text)
            
            #Normalize
            #The loss was trained on normalized vectors, so prediction must use them too
            v_img = F.normalize(v_img, p=2, dim=1, eps=1e-8)
            t_text = F.normalize(t_text, p=2, dim=1, eps=1e-8)
            
            #Calculate Cosine Similarity
            #Dot product of normalized vectors is Cosine Similarity
            
            #WE got the [batch_size] vector shape at the end - similarity for each of the pair
            similarity = (v_img * t_text).sum(dim=1)
            
            #Applying treshold for each of the similarity to get bool value if its match or not
            is_match = (similarity > threshold)
            
            return is_match, similarity    
    
    


    

    
    
    
    
    
    
    
    
    
    
    
    