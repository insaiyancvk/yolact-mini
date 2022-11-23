import torch
import torch.nn.functional as F

from data import cfg, mask_type
# from utils import timer
from .box_utils import crop, sanitize_coordinates

def postprocess(det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
                crop_masks=True, score_threshold=0):
    
    dets = det_output[batch_idx]
    net = dets['net']
    dets = dets['detection']

    if dets is None:
        return [torch.Tensor()] * 4 # Warning, this is 4 copies of the same thing

    if score_threshold > 0:
        keep = dets['score'] > score_threshold

        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]
        
        if dets['score'].size(0) == 0:
            return [torch.Tensor()] * 4
    
    # Actually extract everything from dets now
    classes = dets['class']
    boxes   = dets['box']
    scores  = dets['score']
    masks   = dets['mask']

    if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
        # At this points masks is only the coefficients
        proto_data = dets['proto']

        masks = proto_data @ masks.t()
        masks = cfg.mask_proto_mask_activation(masks)

        # Crop masks before upsampling because you know why
        if crop_masks:
            masks = crop(masks, boxes)

        # Permute into the correct output shape [num_dets, proto_h, proto_w]
        masks = masks.permute(2, 0, 1).contiguous()

        # if cfg.use_maskiou:
            # with timer.env('maskiou_net'):                
        with torch.no_grad():
            maskiou_p = net.maskiou_net(masks.unsqueeze(1))
            maskiou_p = torch.gather(maskiou_p, dim=1, index=classes.unsqueeze(1)).squeeze(1)
            if cfg.rescore_mask:
                if cfg.rescore_bbox:
                    scores = scores * maskiou_p
                else:
                    scores = [scores, scores * maskiou_p]

        # Scale masks up to the full image
        masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False).squeeze(0)

        # Binarize the masks
        masks.gt_(0.5)

    
    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
    boxes = boxes.long()

    if cfg.mask_type == mask_type.direct and cfg.eval_mask_branch:
        # Upscale masks
        full_masks = torch.zeros(masks.size(0), h, w)

        for jdx in range(masks.size(0)):
            x1, y1, x2, y2 = boxes[jdx, :]

            mask_w = x2 - x1
            mask_h = y2 - y1

            # Just in case
            if mask_w * mask_h <= 0 or mask_w < 0:
                continue
            
            mask = masks[jdx, :].view(1, 1, cfg.mask_size, cfg.mask_size)
            mask = F.interpolate(mask, (mask_h, mask_w), mode=interpolation_mode, align_corners=False)
            mask = mask.gt(0.5).float()
            full_masks[jdx, y1:y2, x1:x2] = mask
        
        masks = full_masks

    return classes, scores, boxes, masks