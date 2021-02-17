# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:19:48 2021

@author: 20200016
"""

class Metrics:
    def __init__(self, probabilities, labels, integral='trapezoid'):
        self.probabilities = probabilities
        self.labels = labels
        self.integral = integral
        assert integral in ['trapezoid','max','min'], '"'+str(integral)+'"'+ ' is not a valid integral value. Choose between "trapezoid", "min" or "max"'
        self.probabilities_set = sorted(list(set(probabilities)))
    
    #make predictions based on the threshold value and self.probabilities
    def make_predictions(self, threshold):
        predictions = []
        for prob in self.probabilities:
            if prob >= threshold:
                predictions.append(1)
            else: 
                predictions.append(0)
        return predictions
    
    #make list with kappa scores for each threshold
    def kappa_curve(self):
        kappa_list = []
        
        for thres in self.probabilities_set:
            preds = self.make_predictions(thres)
            tp, tn, fp, fn = self.confusion_matrix(preds)
            k = self.calc_kappa(tp, tn, fp, fn)
            kappa_list.append(k)
        return self.add_zero_to_curve(kappa_list)
    
    #make list with fpr scores for each threshold
    def fpr_curve(self):
        fpr_list = []
        
        for thres in self.probabilities_set:
            preds = self.make_predictions(thres)
            tp, tn, fp, fn = self.confusion_matrix(preds)
            fpr = self.calc_fpr(fp, tn)
            fpr_list.append(fpr)
        return self.add_zero_to_curve(fpr_list)
    
    def tpr_curve(self):
        tpr_list = []
               
        for thres in self.probabilities_set:
            preds = self.make_predictions(thres)
            tp, _, _, fn = self.confusion_matrix(preds)
            tpr = self.calc_tpr(tp, fn)
            tpr_list.append(tpr)
        
        return self.add_zero_to_curve(tpr_list)

    #make list with precision scores for each threshold
    def precision_curve(self):
        precision_list = []
        
        for thres in self.probabilities_set:
            preds = self.make_predictions(thres)
            tp, _, fp, _ = self.confusion_matrix(preds)
            precision = self.calc_precision(tp, fp)
            precision_list.append(precision)
        return self.add_one_to_curve(precision_list)

    #calculate confusion matrix
    def confusion_matrix(self, predictions):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i, pred in enumerate(predictions):
            if pred == self.labels[i]:
                if pred == 1:
                    tp += 1
                else: 
                    tn += 1
            elif pred == 1:
                fp += 1
            else: fn += 1
            tot = tp + tn + fp + fn
        return tp/tot, tn/tot, fp/tot, fn/tot
        
    #Calculate AUK
    def calc_auk(self):        
        auk=0
        fpr_list = self.fpr_curve()
            
        for i, prob in enumerate(self.probabilities_set[:-1]):
            x_dist = abs(fpr_list[i+1] - fpr_list[i])
                
            preds = self.make_predictions(prob) 
            tp, tn, fp, fn = self.confusion_matrix(preds)
            kapp1 = self.calc_kappa(tp, tn, fp, fn)
                
            preds = self.make_predictions(self.probabilities_set[i+1]) 
            tp, tn, fp, fn = self.confusion_matrix(preds)
            kapp2 = self.calc_kappa(tp, tn, fp, fn)
                
            y_dist = abs(kapp2-kapp1)
            bottom = min(kapp1, kapp2)*x_dist
            auk += bottom
            if self.integral is 'trapezoid':
                top = (y_dist * x_dist)/2
                auk += top
            elif self.integral is 'max':
                top = (y_dist * x_dist)
                auk += top
            else:
                continue
        return auk
       
    #Calculate roc-auc
    def calc_roc_auc(self):
        roc_auc = 0
        fpr_list = self.fpr_curve()
        
        for i, prob in enumerate(self.probabilities_set[:-1]):
            x_dist = abs(fpr_list[i+1] - fpr_list[i])
            
            preds = self.make_predictions(prob) 
            tp, _, _, fn = self.confusion_matrix(preds)
            tpr1 = self.calc_tpr(tp, fn)
            
            preds = self.make_predictions(self.probabilities_set[i+1]) 
            tp, tn, fp, fn = self.confusion_matrix(preds)
            tpr2 = self.calc_tpr(tp, fn)

            y_dist = abs(tpr2-tpr1) 
            bottom = x_dist * min(tpr1, tpr2)
            roc_auc += bottom
            
            if self.integral is 'trapezoid':
                top = (y_dist * x_dist)/2
                roc_auc += top
            elif self.integral is 'max':
                top = (y_dist * x_dist)
                roc_auc += top
            else:
                continue
        return roc_auc
    
    def calc_pr_auc(self):
        pr_auc = 0
        tpr_list = self.tpr_curve()
        
        for i, prob in enumerate(self.probabilities_set[:-1]):
            x_dist = abs(tpr_list[i+1] - tpr_list[i])
            
            preds = self.make_predictions(prob) 
            tp, _, fp, _ = self.confusion_matrix(preds)
            precision1 = self.calc_precision(tp, fp)
             
            preds = self.make_predictions(self.probabilities_set[i+1]) 
            tp, _, fp, _ = self.confusion_matrix(preds)
            precision2 = self.calc_precision(tp, fp)

            y_dist = abs(precision2-precision1) 
            bottom = x_dist * min(precision1, precision2)
            pr_auc += bottom
            
            if self.integral is 'trapezoid':
                top = (y_dist * x_dist)/2
                pr_auc += top
            elif self.integral is 'max':
                top = (y_dist * x_dist)
                pr_auc += top
            else:
                continue
        return pr_auc

            #The code below seems unnecessary now that I have added the extra areas in the curve
        '''
            if step is False:                    
                top = (y_dist * x_dist)/2
                pr_auc += top
        
        if step is False:
            #add begin area before smallest probability  
            preds = self.make_predictions(min(self.probabilities_set))
            tp, _, fp, _ = self.confusion_matrix(preds)
            precision = self.calc_precision(tp, fp)
            begin = (precision*min(tpr_list))/2
            pr_auc += begin
        
        #add end area after largest probability 
        preds=self.make_predictions(max(self.probabilities_set))
        tp, _, fp, _ = self.confusion_matrix(preds)
        precision = self.calc_precision(tp, fp)
        y_diff = 1-precision
        x_diff= 1-max(tpr_list)
        end_bottom = precision * x_diff
        pr_auc += end_bottom

        if step is False:
            end_top = (y_diff)*(x_diff)/2
            pr_auc += end_top
        return pr_auc
        '''
        
    def calc_fpr(self, fp, tn):
        return fp/(fp+tn)
    
    def calc_tpr(self, tp, fn): #same as recall
        return tp/(tp+fn)
    
    def calc_precision(self, tp, fp):
        return tp/(tp+fp)

    #Calculate kappa score
    def calc_kappa(self, tp, tn, fp, fn):
        acc = tp + tn
        p = tp + fn
        p_hat = tp + fp
        n = fp + tn
        n_hat = fn + tn
        p_c = p * p_hat + n * n_hat
        return (acc - p_c) / (1 - p_c)    

    #Add zero to appropriate position in list
    def add_zero_to_curve(self, curve):
        min_index = curve.index(min(curve)) 
        if min_index> 0:
            curve.append(0)
        else: curve.insert(0,0)
        return curve
    
        #Add zero to appropriate position in list
    def add_one_to_curve(self, curve):
        max_index = curve.index(max(curve)) 
        if max_index> 0:
            curve.append(1)
        else: curve.insert(0,1)
        return curve