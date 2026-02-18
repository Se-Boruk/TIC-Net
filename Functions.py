import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Custom_loss(nn.Module):
    def __init__(self, margin=0.7, triplet_weight=1.0, contrastive_weight=2.0, init_temp=0.07):
        super().__init__()
        # Margin 0.7 zgodnie z Twoim Configiem
        self.triplet = nn.TripletMarginLoss(margin=margin, p=2)
        
        self.triplet_weight = triplet_weight
        self.contrastive_weight = contrastive_weight
        
        # Logarytm skali dla stabilności numerycznej
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temp))

    def forward(self, v_main, v_aug, t_pos, t_neg):
        # 1. Wyliczamy temperaturę i clampujemy
        with torch.no_grad():
            self.logit_scale.clamp_(0, 4.6052)

        labels = torch.arange(v_main.size(0), device=v_main.device)
        
        # 2. RZUTOWANIE DO FLOAT32 DLA STABILNOŚCI
        # To jest kluczowe przy FP16 training!
        #v_main = v_main.float()
        #v_aug = v_aug.float()
        #t_pos = t_pos.float()
        #t_neg = t_neg.float()
        #s = s.float()

        s = self.logit_scale.exp() 
        v_main, v_aug = v_main.float(), v_aug.float()
        t_pos, t_neg = t_pos.float(), t_neg.float()
        
        # =================================================================
        # 1. SYMMETRIC CONTRASTIVE LOSS
        # =================================================================
        
        # --- A. Main Image <-> Text ---
        logits_main = torch.matmul(v_main, t_pos.T) * s
        loss_i2t_main = F.cross_entropy(logits_main, labels)
        loss_t2i_main = F.cross_entropy(logits_main.T, labels)
        loss_main = (loss_i2t_main + loss_t2i_main) / 2.0

        # --- B. Augmented Image <-> Text ---
        logits_aug = torch.matmul(v_aug, t_pos.T) * s
        loss_i2t_aug = F.cross_entropy(logits_aug, labels)
        loss_t2i_aug = F.cross_entropy(logits_aug.T, labels)
        loss_aug = (loss_i2t_aug + loss_t2i_aug) / 2.0
        
        loss_contrastive = (loss_main + loss_aug) / 2.0

        # =================================================================
        # 2. TRIPLET LOSS
        # =================================================================
        loss_triplet_main = self.triplet(v_main, t_pos, t_neg)
        loss_triplet_aug = self.triplet(v_aug, t_pos, t_neg)

        loss_triplet = (loss_triplet_main + loss_triplet_aug) / 2.0

        return (self.triplet_weight * loss_triplet) + (self.contrastive_weight * loss_contrastive)
               
               
               
               
# --- Threshold Calibration Function ---
def calibrate_threshold(pos_scores, neg_scores):
    """Finds the threshold that maximizes training Balanced Accuracy."""
    best_balanced_acc = 0.0
    best_t = 0.5
    
    # Increase granularity to 101 steps for a more precise decision boundary
    for t in np.linspace(0, 1, 101):
        # True Positives & False Negatives (Matching Pairs)
        tp = (pos_scores > t).sum()
        fn = (pos_scores <= t).sum()
        
        # True Negatives & False Positives (Non-Matching Pairs)
        tn = (neg_scores < t).sum()
        fp = (neg_scores >= t).sum()
        
        # Calculate rates
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Balanced Accuracy is the mean of sensitivity and specificity
        balanced_acc = (recall + specificity) / 2
        
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_t = t
            
    return best_t               
               
               
               
def calculate_metrics(pos_scores, neg_scores, threshold):
    # True Positives: Matching pairs above threshold
    tp = (pos_scores > threshold).sum()
    # False Negatives: Matching pairs below threshold
    fn = (pos_scores <= threshold).sum()
    # True Negatives: Non-matching pairs below threshold
    tn = (neg_scores < threshold).sum()
    # False Positives: Non-matching pairs above threshold
    fp = (neg_scores >= threshold).sum()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (recall + specificity) / 2
    
    return balanced_acc, recall, specificity               
               
               
               
               
               
               
               