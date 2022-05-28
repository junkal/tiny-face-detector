import os
import json
import torch
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from torchvision import transforms
from models.model import DetectionModel
from utils.utils import get_bboxes
from utils.nms import nms

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type = str, required = True, default = None, help="Input image file for inference")
    parser.add_argument("--template_file", type = str, default = "./models/templates.json")
    parser.add_argument("--weight", type = str, default = "./models/checkpoint_50_best.pt", help="The path to the model weight")
    parser.add_argument("--prob_thresh", type=float, default=0.1)
    parser.add_argument("--nms_thresh", type=float, default=0.2)

    return parser.parse_args()


def get_model(model_file, device = None):

    model = None

    if not os.path.exists(model_file):
        print("[get_model]: {} not found, downloading from source".format(model_file))
        # os.system("pip install gdown")
        cwd = os.getcwd()
        os.chdir("models")
        os.system("gdown 1J-zdDdgxDHMCsTfrBHs3KpwJVSq6to4v")
        os.chdir(cwd)

    model = torch.load(model_file, map_location=device)
    print("[get_model]: Model loaded successfully")

    return model


def get_detections(model, img, templates, rf, img_transforms,
                   prob_thresh=0.65, nms_thresh=0.3, scales=(-2, -1, 0, 1), device=None):
    model = model.to(device)
    model.eval()

    dets = np.empty((0, 5))  # store bbox (x1, y1, x2, y2), score

    num_templates = templates.shape[0]

    # Evaluate over multiple scale
    scales_list = [2 ** x for x in scales]

    # convert tensor to PIL image so we can perform resizing
    image = transforms.functional.to_pil_image(img[0])

    min_side = np.min(image.size)

    for scale in scales_list:
        # scale the images
        scaled_image = transforms.functional.resize(image,
                                                    np.int(min_side*scale))

        # normalize the images
        img = img_transforms(scaled_image)

        # add batch dimension
        img.unsqueeze_(0)

        # now run the model
        x = img.float().to(device)

        output = model(x)

        # first `num_templates` channels are class maps
        score_cls = output[:, :num_templates, :, :]
        prob_cls = torch.sigmoid(score_cls)

        score_cls = score_cls.data.cpu().numpy().transpose((0, 2, 3, 1))
        prob_cls = prob_cls.data.cpu().numpy().transpose((0, 2, 3, 1))

        score_reg = output[:, num_templates:, :, :]
        score_reg = score_reg.data.cpu().numpy().transpose((0, 2, 3, 1))

        t_bboxes, scores = get_bboxes(score_cls, score_reg, prob_cls,
                                      templates, prob_thresh, rf, scale)

        scales = np.ones((t_bboxes.shape[0], 1)) / scale
        # append scores at the end for NMS
        d = np.hstack((t_bboxes, scores))

        dets = np.vstack((dets, d))

    # Apply NMS
    keep = nms(dets, nms_thresh)
    dets = dets[keep]

    return dets

def main():
    args = arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[main]: Device Type {}".format(device))

    #check for valid image file extension
    if not args.image.lower().endswith(('.jpg', '.jpeg')):
        print("[main]: Only accept jpg or jpeg file type, got {}".format(args.image))
        exit(1)

    templates = json.load(open(args.template_file))
    templates = np.round_(np.array(templates), decimals=8)
    num_templates = templates.shape[0]

    model = get_model(args.weight, device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([transforms.ToTensor(),normalize])

    # hardcoded receptive field from the original author
    rf = {
        'size': [859, 859],
        'stride': [8, 8],
        'offset': [-1, -1]
    }

    img = Image.open(args.image)
    np_im = np.array(img)
    np_im = np.expand_dims(np_im, 0)

    print("[main]: Inferencing image {}".format(args.image))
    with torch.no_grad():
        dets = get_detections(model,
                              np_im,
                              templates,
                              rf,
                              val_transforms,
                              args.prob_thresh,
                              args.nms_thresh,
                              device=device)

    img_draw = ImageDraw.Draw(img)

    print("[main]: Face Count: {}".format(len(dets)))

    for bbox in dets:
        img_draw.rectangle(bbox[:4].tolist(),
                           outline=ImageColor.getrgb('yellow'),
                           width=2)

    output_img_file = os.path.splitext(args.image)[0] + '_processed.jpg'
    img.save(output_img_file)
    print("[main]: Output file saved at {}".format(output_img_file))


if __name__ == '__main__':
    main()
