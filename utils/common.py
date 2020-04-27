import cPickle as pickle
import sys, time, math
import numpy as np
from sklearn import metrics
from PIL import Image

# format printing
def get_time_str():
    return time.strftime("%Y-%m-%d, %H:%M:%S ", time.localtime((time.time()) ))

def print_info(msg):
    print get_time_str(), msg
    sys.stdout.flush() 


# saving/loading data into pkl
def data_to_pkl(data, file_path):
    print "Saving data to file(%s). "%(file_path)
    with open(file_path, "w") as f:
        pickle.dump(data,f)
        return True

    print "Occur Error while saving..."
    return False

def read_pkl(file_path):
    with open(file_path, "r") as f:
        return pickle.load(f)

# data processing
def image_loader_gray(path):
    return Image.open(path).convert('L')

def image_loader_rgb(path):
    return Image.open(path).convert('RGB')

def l2norm(feature_in_rows):
  nr = feature_in_rows.shape[0]
  dim = feature_in_rows.shape[1]
  norms = np.sqrt(np.sum(feature_in_rows ** 2, 1)) + np.finfo(float).eps
  feature_in_rows /= norms.reshape(-1, 1)
  return feature_in_rows

# point related
def distance(p1,p2):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return math.sqrt(dx*dx+dy*dy)

# face detection/alignment metrics
def area2d(b):
    return (b[:,2]-b[:,0]+1)*(b[:,3]-b[:,1]+1)

def overlap2d(b1, b2):
    xmin = np.maximum( b1[:,0], b2[:,0] )
    xmax = np.minimum( b1[:,2]+1, b2[:,2]+1)
    width = np.maximum(0, xmax-xmin)
    ymin = np.maximum( b1[:,1], b2[:,1] )
    ymax = np.minimum( b1[:,3]+1, b2[:,3]+1)
    height = np.maximum(0, ymax-ymin)   
    return width*height          

def iou2d(b1, b2):
    if b1.ndim == 1: b1 = b1[None,:]
    if b2.ndim == 1: b2 = b2[None,:]
    assert b2.shape[0]==1
    o = overlap2d(b1, b2)
    return o / ( area2d(b1) + area2d(b2) - o ) 

def nms2d(boxes, overlap=0.3):
    # boxes = x1,y1,x2,y2,score
    if boxes.size==0:
        return np.array([],dtype=np.int32)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    scores = boxes[:,4]
    areas = (x2-x1+1)*(y2-y1+1)
    I = np.argsort(scores)
    indices = np.zeros(scores.shape, dtype=np.int32)
    counter = 0
    while I.size>0:
        i = I[-1]
        indices[counter] = i
        counter += 1
        xx1 = np.maximum(x1[i],x1[I[:-1]])
        yy1 = np.maximum(y1[i],y1[I[:-1]])        
        xx2 = np.minimum(x2[i],x2[I[:-1]])
        yy2 = np.minimum(y2[i],y2[I[:-1]]) 
        inter = np.maximum(0.0,xx2-xx1+1)*np.maximum(0.0,yy2-yy1+1)
        iou = inter / ( areas[i]+areas[I[:-1]]-inter)
        I = I[np.where(iou<=overlap)[0]]
    return indices[:counter]  

def det_roc(pred, gt, metric = 'discrete'):
    # pred, gt are as format: N x [im_idx, dets[i].left(), dets[i].top(), dets[i].right(), dets[i].bottom(), scores[i]]
    num_face = gt.shape[0]
    fp = 0
    tp = 0
    tpr = []
    fps = [0]
    sorted_ind = np.argsort(-pred[:, -1])
    for i, k in enumerate(sorted_ind):
        box = pred[k,:]
        ispositive = False
        index_this = np.where(gt[:,0]==box[0])[0] # faces in gt
        if index_this.size>0:
            BBGT = gt[index_this, 1:]
            iou = iou2d(BBGT, box[1:5])
            argmax = np.argmax(iou) # get the max overlap window
            
            if metric=='discrete' and iou[argmax]>=0.5: # discrete
                ispositive = True
                gt = np.delete(gt, index_this[argmax], 0)
        if ispositive:
            tp += 1
        else:
            fp += 1
            fps.append(fp) # actually no need for fps       
        tpr.append(float(tp)/float(tp+fp))

    return (tpr, fps)

def det_pr(pred, gt, iou_thresh = 0.5):
    pr = np.empty((pred.shape[0]+1,2), dtype=np.float32) # precision,recall
    pr[0,0] = 1.0
    pr[0,1] = 0.0
    fn = gt.shape[0]
    fp = 0
    tp = 0
    sorted_ind = np.argsort(-pred[:, -1])
    for i, k in enumerate(sorted_ind):
        box = pred[k,:]
        ispositive = False
        index_this = np.where(gt[:,0]==box[0])[0]
        if index_this.size>0:
            BBGT = gt[index_this, 1:]
            iou = iou2d(BBGT, box[1:5])
            argmax = np.argmax(iou) # get the max overlap window
            if iou[argmax]>=iou_thresh:
                ispositive = True
                gt = np.delete(gt, index_this[argmax], 0)
        if ispositive:
            tp += 1
            fn -= 1
        else:
            fp += 1
        pr[i+1,0] = float(tp)/float(tp+fp)
        pr[i+1,1] = float(tp)/float(tp+fn)
        
    return pr

# face recognition/verfication metrics
'''
open-set face recognition uses ROC with 
    x: false alarm / false positive rate (or rank when the gallery is large)
    y: detection&recognition / true positive / precision / true accept rate

close-set face recognition uses ROC with 
    x: rank
    y: detection&recognition / true positive / precision / true accept rate

verification:
    a lot, e.g., rank-1, eer, hter, farfrr, pr 

Note:
    The following functions follows bob.measure
    and always use neg and pos scores which from neg and pos pairs
'''
def farfrr(negatives, positives, threshold) :
    num_neg = negatives.shape[0]
    num_pos = positives.shape[0]
    false_accepts = np.sum(negatives>=threshold) # false positive
    false_rejects = np.sum(positives<threshold) # false negative
    return false_accepts/float(num_neg), false_rejects/float(num_pos)

def precision_recall(negatives, positives, threshold):
    num_neg = negatives.shape[0]
    num_pos = positives.shape[0]
    true_positives = np.sum(positives>=threshold)
    false_negtives = np.sum(negatives>=threshold)
    total_classified_positives = true_positives + false_positives;
    if total_classified_positives==0: total_classified_positives = 1
    if num_pos==0: num_pos = 1    
    return float(true_positives)/total_classified_positives, total_classified_positives/float(num_pos)

def f_score(negatives, positives, threshold, weight):
    pass


def recognition_rate(cmc_scores, threshold = None, rank = 1):
  """Calculates the recognition rate from the given input

  Parameters:

    cmc_scores (list): A list in the format ``[(negatives, positives), ...]``

      Each pair contains the ``negative`` and the ``positive`` scores for **one
      probe item**.  Each pair can contain up to one empty array (or ``None``),
      i.e., in case of open set recognition.

    threshold (:obj:`float`, optional): Decision threshold. If not ``None``, **all**
      scores will be filtered by the threshold. In an open set recognition
      problem, all open set scores (negatives with no corresponding positive)
      for which all scores are below threshold, will be counted as correctly
      rejected and **removed** from the probe list (i.e., the denominator).

    rank (:obj:`int`, optional):
      The rank for which the recognition rate should be computed, 1 by default.


  Returns:

    float: The (open set) recognition rate for the given rank, a value between
    0 and 1.

  """
  # If no scores are given, the recognition rate is exactly 0.
  if not cmc_scores:
    return 0.

  correct = 0
  counter = 0
  for neg, pos in cmc_scores:
    # set all values that are empty before to None
    if pos is not None and not np.array(pos).size:
      pos = None
    if neg is not None and not np.array(neg).size:
      neg = None

    if pos is None and neg is None:
      raise ValueError("One pair of the CMC scores has neither positive nor negative values")

    # filter out any negative or positive scores below threshold; scores with exactly the threshold are also filtered out
    # now, None and an empty array have different meanings.
    if threshold is not None:
      if neg is not None:
        neg = np.array(neg)[neg > threshold]
      if pos is not None:
        pos = np.array(pos)[pos > threshold]

    if pos is None:
      # no positives, so we definitely do not have a match;
      # check if we have negatives above threshold
      if not neg.size:
        # we have no negative scores over the threshold, so we have correctly rejected the probe
        # don't increase any of the two counters...
        continue
      # we have negatives over threshold, so we have incorrect classifications; independent on the actual rank
      counter += 1
    else:
      # we have a positive, so we need to count the probe
      counter += 1

      if not np.array(pos).size:
        # all positive scores have been filtered out by the threshold, we definitely have a mis-match
        continue

      # get the maximum positive score for the current probe item
      # (usually, there is only one positive score, but just in case...)
      max_pos = np.max(pos)

      if neg is None or not np.array(neg).size:
        # if we had no negatives, or all negatives were below threshold, we have a match at rank 1
        correct += 1
      else:
        # count the number of negative scores that are higher than the best positive score
        index = np.sum(neg >= max_pos)
        if index < rank:
          correct += 1
  print_info('{}/{}'.format(correct,counter))
  return float(correct) / float(counter)

def roc(negatives, positives, bExtra = False):
    num_neg = negatives.shape[0]
    num_pos = positives.shape[0]
    y_true = [1]*num_pos + [0]*num_neg
    y_true = np.array(y_true)
    y_pred = np.vstack((positives[:,np.newaxis], negatives[:,np.newaxis]))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1) # or fpr ~ FAR
    extra = {}
    if bExtra:
      fnr = 1 - tpr 
      tnr = 1 - fpr 
      s = np.max(np.where(tnr > tpr))
      eer_thresholds = thresholds[s]
      if s == len(tpr):
          eer = None 
      else:
          if tpr[s] == tpr[s+1]:
            eer = 1 - tpr[s]
          else:
            eer = 1 - tnr[s]
      extra['EER'] = eer
      extra['EER_threshold'] = eer_thresholds

      # calc tpr @ fpr==0(tnr==1)
      s = np.max(np.where(tnr >= 1))
      if s == len(tpr):
          extra['ACC@FAR_0%'] = None
      else:
          extra['ACC@FAR_0%'] = tpr[s]
          extra['ACC@FAR_0'] = '{}/{}@{}/{}'.format(int(num_pos*tpr[s]), num_pos, 0, num_neg)

      # calc tpr @ fpr==0.01(tnr==0.99)
      s = np.max(np.where(tnr >= 0.99))
      if s == len(tpr):
          extra['ACC@FAR_1%'] = None
      else:
          extra['ACC@FAR_1%'] = tpr[s]
          extra['ACC@FAR_1%_'] = '{}/{}@{}/{}'.format(int(num_pos*tpr[s]), num_pos, int(num_neg*1/100.0), num_neg)     

      # calc tpr @ fpr==0.001
      if num_neg>1000:
        s = np.max(np.where(tnr >= 0.999))
        if s == len(tpr):
            extra['ACC@FAR_0.1%'] = None
        else:
            extra['ACC@FAR_0.1%'] = tpr[s]
            extra['threshold_0.1%'] = thresholds[s]
            extra['ACC@FAR_0.1%_'] = '{}/{}@{}/{}'.format(int(num_pos*tpr[s]), num_pos, int(num_neg*0.1/100.0), num_neg)    

      # calc tpr @ fpr==0.0001
      if num_neg>10000:
        s = np.max(np.where(tnr >= 0.9999))
        if s == len(tpr):
            extra['ACC@FAR_0.01%'] = None
        else:
            extra['ACC@FAR_0.01%'] = tpr[s] 
            extra['ACC@FAR_0.01%_'] = '{}/{}@{}/{}'.format(int(num_pos*tpr[s]), num_pos, int(num_neg*0.01/100.0), num_neg)    

      # calc tpr @ fpr==0.00001
      if num_neg>100000:
        s = np.max(np.where(tnr >= 0.99999))
        if s == len(tpr):
            extra['ACC@FAR_0.001%'] = None
        else:
            extra['ACC@FAR_0.001%'] = tpr[s]
            extra['ACC@FAR_0.001%_'] = '{}/{}@{}/{}'.format(int(num_pos*tpr[s]), num_pos, int(num_neg*0.001/100.0), num_neg)    

      # calc tpr @ fpr==0.000001
      if num_neg>1000000:
        s = np.max(np.where(tnr >= 0.999999))
        if s == len(tpr):
            extra['ACC@FAR_0.0001%'] = None
        else:
            extra['ACC@FAR_0.0001%'] = tpr[s]
            extra['ACC@FAR_0.0001%_'] = '{}/{}@{}/{}'.format(int(num_pos*tpr[s]), num_pos, int(num_neg*0.0001/100.0), num_neg)    
     # plt.plot(fpr, tpr)
     # plt.show()
    return fpr, tpr, thresholds, extra

def eer(negatives, positives):
    fpr, tpr, thresholds, _ = roc(negatives, positives)
    fnr = 1 - tpr 
    tnr = 1 - fpr 
    s = np.max(np.where(tnr > tpr))
    eer_thresholds = thresholds[s]
    if s == len(tpr):
        eer = None 
    else:
        if tpr[s] == tpr[s+1]:
          eer = 1 - tpr[s]
        else:
          eer = 1 - tnr[s]
        
    return eer, eer_thresholds

class weighted_error(object):
    """docstring for weighted_error"""
    def __init__(self, weight):
        super(weighted_error, self).__init__()
        self._weight = weight
        assert weight<=1.0 and weight>=0.0

    def __call__(self, far, frr):
        return self._weight*far + (1 - self._weight)*frr

class eer_predict(object):
    """docstring for eer_predict"""
    def __init__(self):
        super(eer_predict, self).__init__()
    def __call__(self, far, frr):
        return abs(far-frr)
        
def min_hter_threshold(negatives, positives, crit = 'HTER', thr_range=[-1,1]):
    # for normalized similarity measurement 
    if crit=='HTER':
        predicate = weighted_error(0.5)
    elif crit=='EER':
        predicate = eer_predict()
        
    best_threshold = 0
    best_acc = 0
    current_predicate = 1e8
    min_predicate = 1e8
    min_threshold = 1e8
    # bob version    
    far = 1.
    frr = 0.
    far_decrease = 1./negatives.shape[0]
    frr_increase = 1./positives.shape[0]

    negatives = np.sort(negatives)
    positives = np.sort(positives)
    current_threshold = np.minimum(negatives[0], positives[0])
    pos_ind, neg_ind = 0, 0
    while pos_ind!=positives.shape[0] and neg_ind!=negatives.shape[0]:
        current_predicate = predicate(far, frr)
        if current_predicate <= min_predicate:
            min_predicate = current_predicate
            min_threshold = current_threshold
        if positives[pos_ind]>=negatives[neg_ind]:
            current_threshold = negatives[neg_ind]
            neg_ind += 1
            far -= far_decrease
        else:
            current_threshold = positives[pos_ind]
            pos_ind += 1
            frr += frr_increase
         # // increase positive and negative as long as they contain the same value
        while neg_ind!=negatives.shape[0] and negatives[neg_ind]==current_threshold:
            neg_ind += 1
            far -= far_decrease
        while pos_ind!=positives.shape[0] and positives[pos_ind]==current_threshold:
            pos_ind += 1
            frr += frr_increase
        # compute a new threshold based on the center between last and current score, if we are not already at the end of the score lists
        if neg_ind!=negatives.shape[0] or pos_ind!=positives.shape[0]:
            if neg_ind!=negatives.shape[0] and pos_ind!=positives.shape[0]:
                current_threshold += np.minimum(positives[pos_ind], negatives[neg_ind])
            elif neg_ind!=negatives.shape[0]:
                current_threshold += negatives[neg_ind]
            else:
                current_threshold += positives[pos_ind]
            current_threshold /= 2.0

    current_predicate = predicate(far, frr);
    if current_predicate < min_predicate:
        min_predicate = current_predicate
        min_threshold = current_threshold

    best_threshold = min_threshold

    # # simple version
    # for thr in np.arange(thr_range[0],thr_range[1], 0.005):
    #     far, frr = farfrr(negatives, positives, thr)
    #     current_predicate = predicate(far, frr)
    #     if current_predicate<min_predicate:
    #         min_predicate = current_predicate
    #         best_threshold = thr

    return best_threshold

def hter(negatives, positives, threshold):
    far, frr = farfrr(negatives, positives, threshold)
    predicate = weighted_error(0.5)
    return predicate(far, frr)

if __name__ == "__main__":
    pass
