import torch
import itertools
import numpy as np
import hashlib
from PIL import Image, ImageDraw

import util.misc as utils
import datasets.transforms as T

def draw_box(img, bbox, class_label):
    draw = ImageDraw.Draw(img)
    width = 3
    # text_width, text_height = draw.textsize(class_label)
    text_width = 10
    text_height = 10
    color = '#' + hashlib.md5(class_label.encode('utf-8')).hexdigest()[:6]  # 使用类别名称的哈希值作为颜色值
    x1, y1, _, _ =bbox
    draw.rectangle(bbox, outline=color, width=width)
    draw.rectangle([x1, y1, x1+text_width, y1+text_height], fill=color)
    draw.text([x1+width, y1+width], class_label, fill='white')
    return img

@torch.no_grad()
def predict_single(image_path, model, postprocessors, device):
    correct_mat = np.load('/nfs/lby/code/qpic/data/hico_20160224_det/annotations/corre_hico.npy')
    print("-----------in predict_single-------------")
    model.eval()
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms= T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
            ])
    
    imgg = Image.open(image_path).convert('RGB')
    w, h = imgg.size
    orig_size = torch.as_tensor([int(h), int(w)])
    orig_size = torch.unsqueeze(orig_size, dim=0)

    img, _ = transforms(imgg, None)
    img = torch.unsqueeze(img, dim=0)
    img = img.to(device)
    outputs = model(img)

    results = postprocessors['hoi'](outputs, orig_size)
    preds = list(itertools.chain.from_iterable(utils.all_gather(results)))
    # print(preds)
    pred_rs = []
    for img_preds in preds:
        img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
        bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
        hoi_scores = img_preds['verb_scores']
        print("----hoi_scores shape--------", hoi_scores.shape)
        verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
        print("----verb_labels------------", verb_labels)
        print("----verb_labels shape--------", verb_labels.shape)
        subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
        print("----subject_ids --------", subject_ids)
        print("----subject_ids shape--------", subject_ids.shape)
        object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

        hoi_scores = hoi_scores.ravel()
        print("----after ravel--------", hoi_scores.shape)
        verb_labels = verb_labels.ravel()
        subject_ids = subject_ids.ravel()
        print("------subject_ids-------", subject_ids)
        object_ids = object_ids.ravel()

        if len(subject_ids) > 0:
            object_labels = np.array([bboxes[object_id]['category_id'] for object_id in object_ids])
            masks = correct_mat[verb_labels, object_labels]
            hoi_scores *= masks

            hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                    subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores)]
            hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
            hois = hois[:100]
        else:
            hois = []

        pred_rs.append({
                'predictions': bboxes,
                'hoi_prediction': hois
              })
    print("---------------len(pred_rs-------------", len(bboxes))
    
    for pred_r in pred_rs:
        pred_bboxes = pred_r['predictions']
        pred_hois = pred_r['hoi_prediction'] 
        pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        # print(pred_hois)
        sub_id = pred_hois[0]['subject_id']
        obj_id = pred_hois[0]['object_id']

        # sx1, sy1, sx2, sy2 = pred_bboxes[sub_id]['bbox']
        # ox1, oy1, ox2, oy2 = pred_bboxes[obj_id]['bbox']
        # draw.rectangle([sx1, sy1, sx2, sy2], outline='red', width=3)
        # draw.text((sx1, sy1 - 10), pred_hois[0]['category_id'], fill='red')
        # draw.rectangle([ox1, oy1, ox2, oy2], outline='green', width=3)
        # draw.text((ox1, oy1 - 10), pred_bboxes[obj_id]['category_id'], fill='green')
        # imgg = draw_box(imgg, pred_bboxes[sub_id]['bbox'], pred_hois[0]['category_id'])
        # imgg = draw_box(imgg, pred_bboxes[obj_id]['bbox'], pred_bboxes[obj_id]['category_id'])

    imgg.save("./test.jpg")
        
        