import numpy as np


def visualize_seg(label_map, mc):
    out=np.zeros( (label_map.shape[0], label_map.shape[1], label_map.shape[2], 3))

    for l in range(1, mc.NUM_CLASS):
        out[label_map==l, :] = mc.CLS_COLOR_MAP[l]

    return out

def bgr_to_rgb(ims):
  """Convert a list of images from BGR format to RGB format."""
  out = []
  for im in ims:
    out.append(im[:,:,::-1])
  return out


def rmse_fn(diff, nnz):
  return np.sqrt(np.sum(diff**2)/nnz)

def conf_error_rate_at_thresh_fn(mask, conf, thresh):
  return np.mean((conf>thresh) != mask)


def abs_accuracy_at_thresh_fn(diff, thresh, mask):
  return np.sum((np.abs(diff) < thresh)*mask)/float(np.sum(mask))

def rel_accuracy_at_thresh_fn(pred_ogm, gt_ogm, mask, thresh):
  return np.sum(
      mask * (np.maximum(pred_ogm, gt_ogm) / 
              np.minimum(gt_ogm, pred_ogm) < thresh)
      )/float(np.sum(mask))
  
def evaluate(label, pred, n_class):
    """Evaluation script to compute pixel level IoU.

    Args:
        label: N-d array of shape [batch, W, H], where each element is a class index.
        pred: N-d array of shape [batch, W, H], the each element is the predicted class index.
        n_class: number of classes
        epsilon: a small value to prevent division by 0

    Returns:
        IoU: array of lengh n_class, where each element is the average IoU for this class.
        tps: same shape as IoU, where each element is the number of TP for each class.
        fps: same shape as IoU, where each element is the number of FP for each class.
        fns: same shape as IoU, where each element is the number of FN for each class.
    """

    assert label.shape == pred.shape, \
        'label and pred shape mismatch: {} vs {}'.format(label.shape, pred.shape)

    label = label.cpu().numpy()
    pred = pred.cpu().numpy()

    tp = np.zeros(n_class)
    fn = np.zeros(n_class)
    fp = np.zeros(n_class)

    for cls_id in range(n_class):
        tp_cls = np.sum(pred[label == cls_id] == cls_id)
        fp_cls = np.sum(label[pred == cls_id] != cls_id)
        fn_cls = np.sum(pred[label == cls_id] != cls_id)

        tp[cls_id] = tp_cls
        fp[cls_id] = fp_cls
        fn[cls_id] = fn_cls

    return tp, fp, fn

def print_evaluate(mc, name, value):
    print(f'{name}:')
    for i in range(1, mc.NUM_CLASS):
        print(f'{mc.CLASSES[i]}: {value[i]}')
    print()


