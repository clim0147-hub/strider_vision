# --- keep your existing imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# >>> NEW: small utility
def _make_head(in_ch, out_ch):
    # light conv head for dense maps
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1)
    )

class ElevationNet(nn.Module):
    """
    Adds a multi-instance detection head on top of your single-output API.

    Deployment outputs:
      A) nearest hazard  : [N,6]  -> "hazard_predictions" (backward-compatible)
      B) multi detections: [N,K,10] -> "hazard_detections"
         per detection fields = [score, class_id, dist, offset, severity, surface, x1, y1, x2, y2]
         distance_m in meters (0..3), offset in [-1,1], box coords in pixels
    """
    def __init__(self, in_ch: int = 3, pretrained: bool = True, feat_bn: bool = False,
                 use_imagenet_norm: bool = False,  # keep your earlier optional norm if you added it
                 max_dets: int = 20):
        super().__init__()
        self.max_dets = max_dets

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        # (same first-conv adaptation as your file)
        old_conv = backbone.features[0][0]
        if in_ch != old_conv.in_channels:
            new_conv = nn.Conv2d(in_ch, old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding,
                                 bias=False)
            with torch.no_grad():
                copy_ch = min(3, in_ch)
                new_conv.weight[:, :copy_ch] = old_conv.weight[:, :copy_ch]
                if in_ch > 3:
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:in_ch] = mean_w.repeat(1, in_ch-3, 1, 1)
            backbone.features[0][0] = new_conv

        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # GAP branch for your original [N,6]
        self.feat_dim = 576
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.BatchNorm1d(self.feat_dim) if feat_bn else nn.Identity()
        self.head_conf     = nn.Linear(self.feat_dim, 1)
        self.head_type     = nn.Linear(self.feat_dim, 5)
        self.head_distance = nn.Linear(self.feat_dim, 1)
        self.head_offset   = nn.Linear(self.feat_dim, 1)
        self.head_severity = nn.Linear(self.feat_dim, 1)
        self.head_surface  = nn.Linear(self.feat_dim, 1)

        # >>> NEW: dense multi-instance head on the last feature map (stride=32 -> ~6x10 for 192x320)
        C = self.feat_dim  # 576
        # per-class heatmaps (sigmoid), one channel each
        self.det_heatmap   = _make_head(C, 5)     # [N,5,Hf,Wf]
        # objectness/confidence (sigmoid)
        self.det_conf      = _make_head(C, 1)     # [N,1,Hf,Wf]
        # center offsets in cell (dx,dy) in [0,1] each (sigmoid)
        self.det_dxy       = _make_head(C, 2)     # [N,2,Hf,Wf]
        # box sizes (w,h) as fraction of frame in [0,1] each (sigmoid)
        self.det_wh        = _make_head(C, 2)     # [N,2,Hf,Wf]
        # regression maps per instance
        self.det_dist      = _make_head(C, 1)     # [N,1,Hf,Wf] -> 0..3
        self.det_offset    = _make_head(C, 1)     # [N,1,Hf,Wf] -> -1..1
        self.det_severity  = _make_head(C, 1)     # [N,1,Hf,Wf] -> 0..1
        self.det_surface   = _make_head(C, 1)     # [N,1,Hf,Wf] -> 0..1

    def _nearest_single(self, f: torch.Tensor) -> torch.Tensor:
        """Produce your legacy [N,6] single-hazard output from pooled features."""
        f1 = self.pool(f).flatten(1)
        f1 = self.norm(f1)
        eps = 1e-5
        conf      = torch.clamp(torch.sigmoid(self.head_conf(f1)), eps, 1.0-eps)
        type_log  = self.head_type(f1)
        dist_m    = torch.sigmoid(self.head_distance(f1)) * 3.0
        offset    = torch.tanh(self.head_offset(f1))
        severity  = torch.clamp(torch.sigmoid(self.head_severity(f1)), eps, 1.0-eps)
        surface   = torch.clamp(torch.sigmoid(self.head_surface(f1)),  eps, 1.0-eps)
        type_id   = torch.argmax(type_log, dim=1, keepdim=True).float()
        return torch.cat([conf, type_id, dist_m, offset, severity, surface], dim=1)  # [N,6]

    def _decode_multi(self, f: torch.Tensor, frame_w: int = 320, frame_h: int = 192) -> torch.Tensor:
        """
        Decode dense maps to top-K detections sorted by *ascending distance* (nearest first).
        Returns [N,K,10]: [score, class_id, dist, offset, severity, surface, x1,y1,x2,y2]
        """
        N = f.shape[0]
        Hf = f.shape[2]  # ~6
        Wf = f.shape[3]  # ~10
        stride_y = frame_h / Hf
        stride_x = frame_w / Wf

        heat = torch.sigmoid(self.det_heatmap(f))       # [N,5,Hf,Wf]
        conf = torch.sigmoid(self.det_conf(f))          # [N,1,Hf,Wf]
        dxy  = torch.sigmoid(self.det_dxy(f))           # [N,2,Hf,Wf]
        wh   = torch.sigmoid(self.det_wh(f))            # [N,2,Hf,Wf]
        dist = torch.sigmoid(self.det_dist(f)) * 3.0    # [N,1,Hf,Wf]
        offs = torch.tanh(self.det_offset(f))           # [N,1,Hf,Wf]
        sev  = torch.sigmoid(self.det_severity(f))      # [N,1,Hf,Wf]
        surf = torch.sigmoid(self.det_surface(f))       # [N,1,Hf,Wf]

        # combined score per class & cell
        # score = objectness * per-class heat
        score_map = conf * heat                         # [N,5,Hf,Wf]

        # flatten (class, y, x) to a single axis
        score_flat = score_map.view(N, 5, -1)           # [N,5,Hf*Wf]
        topk_scores, topk_idx = torch.topk(score_flat, k=min(self.max_dets, score_flat.shape[-1]), dim=-1)
        # unravel indices -> class c, cell index p -> y = p//Wf, x = p%Wf
        c_idx = torch.arange(5, device=f.device).view(1,5,1).expand_as(topk_idx)  # [N,5,K]
        # gather per class independently, then reshape to [N, 5*K]
        Kc = topk_scores.shape[-1]
        top_scores = topk_scores.reshape(N, -1)         # [N, 5K]
        top_class  = c_idx.reshape(N, -1).float()       # [N, 5K]
        idx_flat   = topk_idx.reshape(N, -1)            # [N, 5K]
        y_idx = (idx_flat // Wf).float()                # [N, 5K]
        x_idx = (idx_flat %  Wf).float()                # [N, 5K]

        # gather regression maps at those positions
        # build [N, 1, Hf, Wf] -> [N, 5K]
        def g1(m):
            return m.view(N, 1, Hf*Wf).expand(N, 5, Hf*Wf).reshape(N, -1).gather(1, idx_flat)
        def g2(m):  # for 2-ch maps
            m = m.view(N, 2, Hf*Wf).permute(0,2,1)      # [N,Hf*Wf,2]
            gathered = m.gather(1, idx_flat.unsqueeze(-1).expand(-1,-1,2))  # [N,5K,2]
            return gathered

        gx = g1(dxy[:,0:1])    # [N,5K]
        gy = g1(dxy[:,1:2])
        gw = g1(wh[:,0:1])
        gh = g1(wh[:,1:2])
        gdist = g1(dist)
        goffs = g1(offs)
        gsev  = g1(sev)
        gsurf = g1(surf)

        # centers in pixels
        cx = (x_idx + gx).clamp(0, Wf-1) * stride_x
        cy = (y_idx + gy).clamp(0, Hf-1) * stride_y

        # box size in pixels (fraction * frame dims)
        bw = gw.clamp(0, 1) * frame_w
        bh = gh.clamp(0, 1) * frame_h
        x1 = (cx - bw/2).clamp(0, frame_w-1)
        y1 = (cy - bh/2).clamp(0, frame_h-1)
        x2 = (cx + bw/2).clamp(0, frame_w-1)
        y2 = (cy + bh/2).clamp(0, frame_h-1)

        # stack fields: [score, class_id, dist, offset, severity, surface, x1,y1,x2,y2]
        dets = torch.stack([
            top_scores, top_class, gdist, goffs, gsev, gsurf, x1, y1, x2, y2
        ], dim=-1)  # [N,5K,10]

        # sort by ascending distance (nearest first)
        order = torch.argsort(dets[:,:,2], dim=-1, descending=False)
        batch_ids = torch.arange(N, device=f.device).unsqueeze(-1)
        dets_sorted = dets[batch_ids, order]

        # keep first K total (cap)
        K = min(self.max_dets, dets_sorted.shape[1])
        dets_out = dets_sorted[:, :K, :]
        return dets_out

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        # backbone features
        f = self.backbone.features(x)    # [N,576,H/32,W/32]

        # A) legacy single hazard [N,6]
        single_out = self._nearest_single(f)

        # B) multi-instance top-K [N,K,10] decoded to pixel coords
        dets_out = self._decode_multi(f, frame_w=x.shape[-1], frame_h=x.shape[-2])

        if return_logits:
            # keep your original type logits for training the single-head CE if you need it
            type_log = self.head_type(self.pool(f).flatten(1))
            return single_out, type_log, dets_out
        return single_out, None, dets_out
