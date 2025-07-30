import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

"""def eval_classification(task, coco_anns, preds, img_ann_dict, mask_path):
    
    breakpoint()
    classes = coco_anns[f'{task}_categories']
    num_classes = len(classes)
    bin_labels = np.zeros((len(coco_anns["annotations"]), num_classes))
    bin_preds = np.zeros((len(coco_anns["annotations"]), num_classes))
    evaluated_frames = []
    bar = tqdm(total=len(coco_anns["annotations"]))
    for idx, ann in enumerate(coco_anns["annotations"]):
        ann_class = int(ann[task])
        bin_labels[idx, :] = label_binarize([ann_class], classes=list(range(0, num_classes)))

        if  ann["image_name"] in preds.keys():
            these_probs = preds[ann["image_name"]]['{}_score_dist'.format(task)]
            if len(these_probs) == 0:
                print("Prediction not found for image {}".format(ann["image_name"]))
                these_probs = np.zeros((1, num_classes))
            else:
                evaluated_frames.append(idx)

            bin_preds[idx, :] = these_probs
        else:
            print("Image {} not found in predictions lists".format(ann["image_name"]))
    
        bar.update(1)
            
    bin_labels = bin_labels[evaluated_frames]
    bin_preds = bin_preds[evaluated_frames]
    
    precision = {}
    recall = {}
    threshs = {}
    ap = {}
    for c in range(0, num_classes):
        precision[c], recall[c], threshs[c] = precision_recall_curve(bin_labels[:, c], bin_preds[:, c])
        ap[c] = average_precision_score(bin_labels[:, c], bin_preds[:, c])

    mAP = np.nanmean(list(ap.values()))
    
    cat_names = [f"{cat['name']}-AP" for cat in classes]
    
    return mAP, dict(zip(cat_names,list(ap.values())))"""

def eval_classification(task, coco_anns, preds, img_ann_dict, mask_path):

    annots = coco_anns['annotations']
    true_labels = []
    predicted_probabilities = []
    for annot in annots:
        frame_name = annot['image_name']
        annot_value = np.array(annot['cvs'])
        pred_probs_value = np.array(preds[frame_name]['cvs_score_dist'])

        true_labels.append(annot_value)
        predicted_probabilities.append(pred_probs_value)

    true_labels = np.array(true_labels)
    predicted_probabilities = np.array(predicted_probabilities)

    average_precisions = []
    for class_idx in range(true_labels.shape[1]):
        class_true = true_labels[:, class_idx]
        class_scores = predicted_probabilities[:, class_idx]
        average_precision = average_precision_score(class_true, class_scores)
        average_precisions.append(average_precision)

    # Calculate the mean of the average precisions across all classes to obtain mAP
    mAP = np.mean(average_precisions)
    C1_ap = average_precisions[0]
    C2_ap = average_precisions[1]
    C3_ap = average_precisions[2]
    
    cat_names = ['C1', 'C2', 'C3']
    ap = [C1_ap, C2_ap, C3_ap]
    return mAP, dict(zip(cat_names, ap))


def eval_precision(task, coco_anns, preds, img_ann_dict, mask_path):
    classes = coco_anns[f'{task}_categories']
    num_classes = len(classes)

    num_labels = np.zeros((len(coco_anns["annotations"])))
    num_preds = np.zeros((len(coco_anns["annotations"])))


    evaluated_frames = []
    bar = tqdm(total=len(coco_anns["annotations"]))
    for idx, ann in enumerate(coco_anns["annotations"]):
        ann_class = int(ann[task])

        num_labels[idx] = ann_class

        if  ann["image_name"] in preds.keys():
            these_probs = preds[ann["image_name"]]['{}_score_dist'.format(task)]
            if len(these_probs) == 0:
                print("Prediction not found for image {}".format(ann["image_name"]))
                these_probs = np.zeros((1, num_classes))
            else:
                evaluated_frames.append(idx)
            num_preds[idx] = np.argmax(these_probs)
        else:
            print("Image {} not found in predictions lists".format(ann["image_name"]))
            breakpoint()
        bar.update(1)
    
    precision =  precision_score(num_labels, num_preds, average=None)
    mprecision = np.nanmean(precision)

    recall = recall_score(num_labels, num_preds, average=None)
    mrecall = np.nanmean(recall)

    msummary = {i: precision[i] for i in range(len(precision))}
    msummary[len(msummary) + 1] = mprecision
    msummary[len(msummary) + 2] = mrecall


    fscore = 2 * (mprecision * mrecall) / (mprecision + mrecall)
    
    cat_names = [f"{cat['name']}" for cat in classes]
    cat_names.append("mP")
    cat_names.append("mR")
    
    return fscore, dict(zip(cat_names,list(msummary.values())))