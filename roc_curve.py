
import sys
import tensorflow as tf
 
class tf_roc():
    def __init__(self, predict_label_file, threshold_num, save_dir):
        '''file format: dataid,predict_score,label
        the predict_score should be between [0, 1]
        the label should be {0 , 1}
        threshold_num: number of threshold will plot'''
        self.threshold_num = threshold_num
        self.trues = 0 #total of True labels 
        self.fpr = [] #false positive
        self.tpr = [] #true positive
        self.ths = [] #thresholds
        self.save_dir = save_dir
        self.writer = tf.train.SummaryWriter(self.save_dir)
 
        #load predict_label_file to predicts and labels
        fd = open(predict_label_file)
        fdl = fd.readline()
        self.predicts = []
        self.labels = []
        self.total = 0
        while len(fdl) > 0:
            val = fdl.split(',')
            self.predicts.append(float(val[1])) 
            self.labels.append(True if int(val[2]) == 1 else False) 
            fdl = fd.readline()
            self.total += 1
        fd.close()

    def calc(self):
        for label in self.labels:
            if label:
                self.trues += 1
        threshold_step = 1. / self.threshold_num
        for t in range(self.threshold_num + 1):
            th = 1 - threshold_step * t # from high to low
            tn, tp, fp, fpr, tpr = self._calc_once(th)
            self.fpr.append(fpr)
            self.tpr.append(tpr)
            self.ths.append(th)
            self._save(fpr, tpr)
        print self.fpr
        print self.tpr
        print self.ths
 
    def _save(self, fpr, tpr):
        summt = tf.Summary()
        summt.value.add(tag="roc", simple_value = tpr)
        self.writer.add_summary(summt, fpr * 100) #for tensorboard step drawable
        self.writer.flush()
 
    def _calc_once(self, t):
        fp = 0
        tp = 0
        tn = 0

        for i in range(self.total):
            if not self.labels[i]: # label False
                if self.predicts[i] >= t: # predict Postive 
                    fp += 1
                else: # predict Negtive 
                    tn += 1
            elif self.predicts[i] >= t: # label True
                tp += 1 # predict Postive

        tpr = tp / float(self.trues)
        #fpr = fp / float(fp + tn) #precision
        fpr = fp / float(fp + tp) #detection
        return tn, tp, fp, fpr, tpr
 
if __name__ == '__main__':
    predict_label_file, threshold_num, save_dir = sys.argv[1:4]
    roc = tf_roc(predict_label_file, int(threshold_num), save_dir)
    roc.calc()
