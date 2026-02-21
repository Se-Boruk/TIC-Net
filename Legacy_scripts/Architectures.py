import torch
import torch.nn as nn
import torch.nn.functional as F





class Image_encoder(nn.Module):
    """
    Image encoder for extraction of feature vector from image to compare 
    it with the feature vector from the text.
    
    It uses the few main modules:
        1)
        InvertedResidual: 
        Its conv block from Mobilenet 2 which uses the depthwise conv.
        This is block of given length and between each smaller block tehre is skip connection.
        
        2)
        Transition block:
        It is just downsample block. it is added for simplicity and compatibility of the Inverted blocks
        #In this block we are changing n of filters and thanks to that the residual block can 
        operate on single filter number and be simplier
        
        3)
    
        Initial and end conv layers followed by the dense layer

    """

    
    class InvertedResidual(nn.Module):
        """
        1x1 Expansion -> 3x3 Depthwise -> 1x1 Projection.
        """
        def __init__(self, channels, depth, expansion=4, dropout_rate=0.1):
            super().__init__()
            hidden_dim = channels * expansion
            self.layers = nn.ModuleList()
            
            for _ in range(depth):
                self.layers.append(nn.Sequential(
                    #1 Conv + Expansion
                    nn.Conv2d(channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU(0.02, inplace=True),
                    
                    #2 Depthwise
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU(0.02, inplace=True),
                    
                    #3 Conv + collapse
                    nn.Conv2d(hidden_dim, channels, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.Dropout2d(dropout_rate)
                ))

        def forward(self, x):
            for layer in self.layers:
                x = x + layer(x) #Skipp connection between tha smaleler blocks 
            return x

    class TransitionBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.02, inplace=True)
            )

        def forward(self, x):
            return self.downsample(x)


    #Start of the init of the network
    def __init__(self, embed_dim, base_filter=32):
        super().__init__()
        
        #Start first network. Initial fast downsample as well
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_filter, 5, padding=2, stride=2, bias=False), 
            nn.BatchNorm2d(base_filter),
            nn.LeakyReLU(0.02)
        )

        # Config: (depth, filter_multiplier) For the residual blocks
        
        configs = [(2, 2), (4, 4), (6, 6), (6, 8), (2, 12)]
        self.stages = nn.ModuleList()
        curr_c = base_filter
        
        #Applying the depthwise network part
        for depth, mult in configs:
            out_c = base_filter * mult
            self.stages.append(self.InvertedResidual(curr_c, depth))
            self.stages.append(self.TransitionBlock(curr_c, out_c))
            curr_c = out_c

        #Creating one dim vector for the fc layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        
        #Fully connected layers: producing the embedded image vector
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(curr_c, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.15),
            nn.Linear(embed_dim, embed_dim)
        )


    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            
        return self.fc(self.gap(x))



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
            nn.BatchNorm1d(embed_dim),
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
        v_main = F.normalize(v_main, p=2, dim=1)
        v_aug  = F.normalize(v_aug, p=2, dim=1)
        t_pos  = F.normalize(t_pos, p=2, dim=1)
        t_neg  = F.normalize(t_neg, p=2, dim=1)

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
            v_img = F.normalize(v_img, p=2, dim=1)
            t_text = F.normalize(t_text, p=2, dim=1)
            
            #Calculate Cosine Similarity
            #Dot product of normalized vectors is Cosine Similarity
            
            #WE got the [batch_size] vector shape at the end - similarity for each of the pair
            similarity = (v_img * t_text).sum(dim=1)
            
            #Applying treshold for each of the similarity to get bool value if its match or not
            is_match = (similarity > threshold)
            
            return is_match, similarity    
    
    


    

    
    
    
    
    
    
    
    
    
    
    
    