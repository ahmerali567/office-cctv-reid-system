"""
reid_model.py – OSNet feature extractor for deep Re-ID (512-dim)
"""
import torch
import torchreid
import numpy as np
import cv2

class OSNetReID:
    def __init__(self, model_name='osnet_x1_0', device='cpu'):
        self.device = torch.device(device)
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            loss='softmax',
            pretrained=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        # remove classification head for feature extraction
        self.model.classifier = torch.nn.Identity()
        
    def extract(self, crop_bgr):
        """
        crop_bgr: numpy array (H,W,3) in BGR order (OpenCV format)
        returns: 512-dim feature vector (L2-normalized)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        # BGR -> RGB, resize to 256x128 (standard for OSNet)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (128, 256))
        img = torch.from_numpy(img).float().permute(2,0,1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img = (img - mean) / std
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(img)
        feat = feat.cpu().numpy().flatten()
        norm = np.linalg.norm(feat) + 1e-8
        return feat / norm