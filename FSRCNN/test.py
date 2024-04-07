import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from FSRCNN.models import FSRCNN
from FSRCNN.utils import convert_ycbcr_to_rgb, preprocess, calc_psnr


from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
print(FILE)
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# if __name__ == '__main__':
def test(
    weight = '3',
    image_file = '',
    scale = 3,
    save_dir = '',
):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights-file', type=str, required=True)
    # parser.add_argument('--image-file', type=str, required=True)
    # parser.add_argument('--scale', type=int, default=3)

    # save_dir = str(ROOT / 'runs/detect/exp' + exp + '/FSRCNN' ) + os.sep
    save_dir = str(str(save_dir) + os.sep)
    os.makedirs(str(save_dir), exist_ok=True)
    # print("save_dir: "+ str(save_dir))
    # print("weight: "+ str(weight))
    # print(type(weight))
    weights_file = ROOT / f'weights/fsrcnn_x{weight}.pth'
    # print(weights_file)
    # print("image_file: " + str(image_file))
    fileName = image_file.split(os.sep)[-1]

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FSRCNN(scale_factor=scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(image_file).convert('RGB')

    image_width = (image.width // scale) * scale
    image_height = (image.height // scale) * scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // scale, hr.height // scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
    # fileNameBic = fileName.replace('.', '_bicubic_x{}.'.format(scale))
    # bicubic.save(str(save_dir) + fileNameBic)

    lr, _ = preprocess(lr, device)
    hr, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)

    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)

    psnr = calc_psnr(hr, preds)
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    
    fileNameFSR = fileName.replace('.', '_fsrcnn_x{}.'.format(scale))
    # increment_path(save_dir,exist_ok=True, mkdir=True)
    print(str(save_dir) + fileNameFSR)
    output.save(str(save_dir) + fileNameFSR)