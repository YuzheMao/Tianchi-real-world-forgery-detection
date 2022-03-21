import os
import math
import torch
import random
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
# from common.transforms import direct_val
import pdb
import cv2
import albumentations as albu
debug = 0


transform_pil = transforms.Compose([
    transforms.ToPILImage(),
])


def inference_single(img, model, th=0):
    model.eval()
    with torch.no_grad():
        img = img.reshape((-1, img.shape[-3], img.shape[-2], img.shape[-1]))
        img = direct_val(img)
        img = img.cuda()
        
        seg = model(img)
        seg = torch.sigmoid(seg).detach().cpu()
        
        seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]

        if len(seg) != 1:
            pdb.set_trace()
        else:
            fake_seg = seg[0]
        if th == 0:
            return fake_seg, max_score
        fake_seg = 255.0 * (fake_seg > 255 * th)
        fake_seg = fake_seg.astype(np.uint8)

    return fake_seg


def random_crop(input_img, crop_h, crop_w):
    y = 0 if input_img.shape[0] == crop_h else np.random.randint(0, input_img.shape[0] - crop_h)
    x = 0 if input_img.shape[1] == crop_w else np.random.randint(0, input_img.shape[1] - crop_w)
    if len(input_img.shape) == 3:
        input_img = input_img[y:y+crop_h, x:x+crop_w, :]
    elif len(input_img.shape) == 2:
        input_img = input_img[y:y+crop_h, x:x+crop_w]
    return input_img


def crop4(img):
    weight, height = img.shape[0], img.shape[1]
    crop1 = img[0:weight//2, 0:height//2]
    crop2 = img[0:weight//2, height//2+1:height]
    crop3 = img[weight//2:weight, 0:height//2]
    crop4 = img[weight//2:weight, height//2:height]
    return crop1, crop2, crop3, crop4


def rand_bbox(size, size2):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
        
    cut_rat_w = random.random()*0.1 + 0.05
    cut_rat_h = random.random()*0.1 + 0.05

    cut_w = int(W * cut_rat_w)
    cut_h = int(H * cut_rat_h)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, min(W, size2[-2]))
    bby1 = np.clip(cy - cut_h // 2, 0, min(H, size2[-1]))
    bbx2 = np.clip(cx + cut_w // 2, 0, min(W, size2[-2]))
    bby2 = np.clip(cy + cut_h // 2, 0, min(H, size2[-1]))

    return bbx1, bby1, bbx2, bby2


def copy_move(img: np.array, img2: np.array, msk: np.array):
    # resize = albu.Resize(512,512)(image=img, mask=msk)
    img = torch.from_numpy(img).permute(2, 0, 1)
    msk = torch.from_numpy(msk)
    size = img.size()
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    if img2 is None:
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), [1024, 1024])

        x_move = random.randrange(-bbx1, (W - bbx2))
        y_move = random.randrange(-bby1, (H - bby2))
        
        img[:, bbx1+x_move:bbx2+x_move, bby1+y_move:bby2+y_move] = img[:, bbx1:bbx2, bby1:bby2]
        msk[bbx1+x_move:bbx2+x_move, bby1+y_move:bby2+y_move] = torch.ones_like(msk[bbx1:bbx2, bby1:bby2])
        img = img.numpy().transpose(1,2,0)
        # img = cv2.rectangle(img.numpy().transpose(1,2,0), pt1=(bby1+y_move, bbx1+x_move), pt2=(bby2+y_move, bbx2+x_move), color=(255,0,0), thickness=5)
    else:
        # resize = albu.Resize(512,512)(image=img2)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
        
        # assert img.shape == img2.shape
        
        bbx1, bby1, bbx2, bby2 = rand_bbox(img2.size(), img.size())

        x_move = random.randrange(-bbx1, (W - bbx2))
        y_move = random.randrange(-bby1, (H - bby2))
           
        img[:, bbx1+x_move:bbx2+x_move, bby1+y_move:bby2+y_move] = img2[:, bbx1:bbx2, bby1:bby2]
        msk[bbx1+x_move:bbx2+x_move, bby1+y_move:bby2+y_move] = torch.ones_like(msk[bbx1:bbx2, bby1:bby2])
        img = img.numpy().transpose(1,2,0)
        # img = cv2.rectangle(img.numpy().transpose(1,2,0), pt1=(bby1+y_move, bbx1+x_move), pt2=(bby2+y_move, bbx2+x_move), color=(255,0,0), thickness=3)
        print()
    return img, msk.numpy()


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)


def splicing(image: np.array, mask: np.array):
    mask_ = np.zeros(image.shape[:2], dtype="uint8")
    
    height = random.randrange(10, image.shape[1] // 10)
    width = random.randrange(1, image.shape[0])
    start_x = random.randrange(0, image.shape[0] - width)
    start_y = random.randrange(0, image.shape[1] - height)
    
    
    x0, y0 = (start_x + width, start_y)
    x1, y1 = (start_x, start_y)
    x2, y2 = (start_x, start_y + height)
    x3, y3 = (start_x + width, start_y + height)

    # index = image_path.split('/')[-1].split('.')[0]

    mask[y1:y2, x1:x0] = 1

    x_mid0, y_mid0 = int((x1 + x2) / 2), int((y1 + y2) / 2)
    x_mid1, y_mi1 = int((x0 + x3) / 2), int((y0 + y3) / 2)

    thickness = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    # thickness = abs(y2 - y1)
    cv2.line(mask_, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
    img = cv2.inpaint(image, mask_, 7, cv2.INPAINT_NS)
    # img = cv2.rectangle(img, pt1=[x2,y1], pt2=[x3,y2], color=(255,0,0), thickness=3)
    return img, mask


def random_ps(image, image2, mask):
    copy_move1 = 0
    copy_move2 = 0
    splicing_tag = 0
    while(copy_move1 == copy_move2 == splicing_tag == 0):
        if random.random() > 0:
            image, mask = copy_move(image, None, mask)
            copy_move1 += 1
        if random.random() > 0:
            image2 = cv2.imread(image2)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            image, mask = copy_move(image, image2, mask)
            copy_move2 += 1
        if random.random() > 0:
            image, mask = splicing(image, mask)
            splicing_tag += 1
    return image, mask


if __name__ == '__main__':
    image = cv2.imread('../data/train/img/1.jpg')
    mask = cv2.imread('../data/train/mask/1.png', cv2.IMREAD_GRAYSCALE)/255
    crop4(image)
    crop4(mask)
