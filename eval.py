import torch
from math import sqrt
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch, sys, cv2
import torch.backends.cudnn as cudnn

from nutils.yolact import Yolact
from nutils.output_utils import postprocess
from nutils.config import cfg


def calc_size_preserve_ar(img_w, img_h, max_size):
    ratio = sqrt(img_w / img_h)
    w = max_size * ratio
    h = max_size / ratio
    return int(w), int(h)

class FastBaseTransform(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.mean = torch.Tensor((103.94, 116.78, 123.68)).float().cuda()[None, :, None, None]
        self.std  = torch.Tensor( (57.38, 57.12, 58.40) ).float().cuda()[None, :, None, None]
        self.transform = cfg.backbone.transform

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std  = self.std.to(img.device)
        
        # img assumed to be a pytorch BGR image with channel order [n, h, w, c]
        if cfg.preserve_aspect_ratio:
            _, h, w, _ = img.size()
            img_size = calc_size_preserve_ar(w, h, cfg.max_size)
            img_size = (img_size[1], img_size[0]) # Pytorch needs h, w
        else:
            img_size = (cfg.max_size, cfg.max_size)

        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, img_size, mode='bilinear', align_corners=False)

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255
        
        if self.transform.channel_order != 'RGB':
            raise NotImplementedError
        
        img = img[:, (2, 1, 0), :, :].contiguous()

        return img

def evalimage(net:Yolact, path:str):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)
    h, w, _ = frame.shape
    t = postprocess(preds, w, h, score_threshold = 0.15)
    idx = t[1].argsort(0, descending=True)[:15]
    masks = t[3][idx]
    np.save('mask_data', masks.cpu().numpy())

def evaluate(net:Yolact):
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False
    cfg.mask_proto_debug = False
    inp = sys.argv[1]
    evalimage(net, inp)
    return

if __name__ == '__main__':

    with torch.no_grad():
        
        if torch.cuda.is_available():
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        print('Generating masks...', end='')
        net = Yolact()
        net.load_weights("yolact_base_54_800000.pth")
        net.eval()

        if torch.cuda.is_available():
            net = net.cuda()

        evaluate(net)
        print(' Done.')