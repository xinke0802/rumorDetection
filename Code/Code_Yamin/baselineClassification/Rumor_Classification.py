"""
Rumor Classification Experiment
author: Yamin
Fall 2015

"""
from time import time
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.svm import SVC

import random
import sys
import numpy as np
import phase1
print(__doc__)



data_root = "../Data/Phase1/"

def plot_confusion_matrix(cm, target_names, title, cmap=plt.cm.Blues):
    plt.imshow(cm.T, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    

def print_results(yPred_test,y_test,target_names,model_name):

    print (y_test)
    print (yPred_test)


    precision=precision_score(y_test, yPred_test, average='binary')
    accuracy=accuracy_score(y_test, yPred_test)
    recall=recall_score(y_test, yPred_test)

    print("Precision: ", precision)
    print("Accuracy: ",accuracy)
    print("Recall: ",recall)

    print("Classification Report: \n", classification_report(y_test.tolist(), yPred_test.tolist(), target_names=target_names))
    print("Confusion Mat: \n", confusion_matrix( y_test.tolist(), yPred_test.tolist(), labels=range(len(target_names))))

    title_model=model_name

    # Compute confusion matrix
    cm = confusion_matrix(y_test, yPred_test)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm,target_names,title=title_model+ ": Confusion matrix")

    return precision,accuracy,recall

def experiment(feat_train,y_train,feat_test,y_test, best_est_model,target_names,model_name):
    clf_train=best_est_model.fit(feat_train,y_train)
    yPred_test = clf_train.predict(feat_test)

    
    precision,accuracy,recall=print_results(yPred_test,y_test,target_names,model_name)

    return precision,accuracy,recall

def get_features(data_name):

    if data_name=='iris':
        target_names=['Setosa', 'Versicolour',  'Virginica']
        iris = datasets.load_iris()
        feat_train_all, y_train = np.array(iris.data), np.array(iris.target)
        
        feat_train_all, feat_test_all, y_train, y_test = cross_validation.train_test_split( feat_train_all, y_train, test_size=0.4, random_state=0)
    else:

        target_names=['non-rumor', 'rumor']
        
        file_name='train_palin_boston_nonrumors.npy'

        feat_train_all= np.load(data_root+file_name)
        feat_test_all=np.load(data_root+"test_michelle_cell_airfrance_obama.npy")
        
        n_train_samples=feat_train_all.shape[0]
        n_test_samples=feat_test_all.shape[0]   
        print("n_train_samples: %d" % n_train_samples)  
        print("n_test_samples: %d" % n_test_samples)
    
        n_features= feat_train_all.shape[1] #exclude the ground truth in feat mat   
        print("n_features: %d" % n_features)
        
        y_train=feat_train_all[:,n_features-1]
        y_test= feat_test_all[:,n_features-1]
        
        feat_train_all=feat_train_all[:,:n_features-1]
        feat_test_all=feat_test_all[:,:n_features-1]

        print '\ntraining set:'
        n_rumor=np.sum(y_train)
        n_samples=y_train.shape[0]
        print "#non_rumor", n_samples-n_rumor
        print "#rumor", n_rumor
        print "ratio of rumor", n_rumor*1.0/n_samples

        print '\ntest set:'
        n_rumor=np.sum(y_test)
        n_samples=y_test.shape[0]
        print "#non_rumor", n_samples-n_rumor
        print "#rumor", n_rumor
        print "ratio of rumor", n_rumor*1.0/n_samples
        
    n_classes=len(target_names)
    print("\nn_classes: %d" % n_classes)

    return target_names, feat_train_all, feat_test_all, y_train, y_test, n_train_samples, n_test_samples,n_classes,n_features

def select_baseline_features(baseline_name,feat_train_all,feat_test_all):
    #build baseline feature matrix using only some feature columns
    ind_all= range(feat_train_all.shape[1])
    ind_pop =[2,3,4]
    ind_qmarks=[0,1,5]

    if baseline_name=='QMarks':
        ind_val=ind_qmarks
    elif baseline_name=='Popularity':
        ind_val=ind_pop 
    elif baseline_name=='All':
        ind_val=ind_all

    feat_train= feat_train_all[:,ind_val]     
    feat_test=feat_test_all[:,ind_val]

    return feat_train, feat_test
    
def main():

    ###############################################################################
    ## STEP 1 - LOAD DATA ##
    
    #Feat Matrix ( n_samples x n_features + 1 <-ground truth)
    #load data
    data_name='iris'
    target_names, feat_train_all, feat_test_all, y_train, y_test, n_train_samples, n_test_samples,n_classes,n_features=get_features(data_name)

    model_list=['forest','SVC'] #,'logReg'

    #baseline_list=['QMarks', 'Popularity','All']
    baseline_list=['All']
    
    ###############################################################################
    ## STEP 2 - Split into a training set and a validation set using a stratified k fold

    best_estimator_list=[]

    C_range=np.array([1e3, 5e3, 1e4, 5e4, 1e5])
    gamma_range=np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1])
    
    trainT_mat=np.zeros(len(model_list))
    testT_mat=np.zeros(len(model_list))
    accuracy_mat=np.zeros(len(model_list))
    precision_mat=np.zeros(len(model_list))
    recall_mat=np.zeros(len(model_list))

    for i in range(len(model_list)):

        print '\ni',i

        ###############################################################################
        # Select features for each baseline model
   
        feat_train, feat_test=feat_train_all,feat_test_all

        #baseline_name=baseline_list[i]
        #feat_train, feat_test=select_baseline_features(baseline_name,feat_train_all,feat_test_all)
        
        ###############################################################################
        # Select Hyperparameter

        print("Grid-searching the best estimator to the training set")      

        model_name=model_list[i]#model_list[i]

        print "MODEL: ", model_name
        
        if model_name=="SVC":
            model=SVC(class_weight='auto')
            param_grid = {'C': C_range,
                  'gamma': gamma_range, }
            clf = GridSearchCV(model, param_grid)
            t0 = time()
            clf.fit(feat_train,y_train)
            trainT_mat[i]=(time() - t0)
            best_estimator_list=best_estimator_list+ [clf.best_estimator_]
                    
        elif model_name=="forest":
            clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=1, random_state=0)
            t0 = time()
            clf.fit(feat_train,y_train)
            trainT_mat[i]=(time() - t0)
            best_estimator_list=best_estimator_list+ [clf]
            
        elif model_name=='logReg':
            model = LogisticRegression(class_weight='auto')
            param_grid = {'C': C_range}
            clf = GridSearchCV(model, param_grid)

            t0 = time()
            clf.fit(feat_train,y_train)
            best_estimator_list=best_estimator_list+ [clf.best_estimator_]
            trainT_mat[i]=(time() - t0)          

        best_model=best_estimator_list[i]

        t0=time()
        ypred_train_temp=clf.predict(feat_train)
        testT_mat[i]=(time() - t0)

        ###############################################################################
        # Use best estimator for prediction test data

        precision,accuracy,recall=experiment(feat_train,y_train,feat_test,y_test,best_model,target_names,model_name)

        precision_mat[i]=precision 
        accuracy_mat[i]=accuracy 
        recall_mat[i]=recall 

    result_mat= np.vstack((trainT_mat,testT_mat))
    param_list=['trainT','testT']
    title="Comparison between classifiers in terms of training and testing times"
    log=1
    ylabel='log seconds'
    plot_bar(model_list,result_mat,title,param_list,'upper left',log,ylabel)

    result_mat=np.vstack((accuracy_mat,precision_mat,recall_mat))
    param_list=['accuracy','precision','recall']
    title="Comparison between classifiers in terms of accuracy, precision, recall"
    log=0
    ylabel='score'
    plot_bar(model_list,result_mat,title,param_list,'upper left',log,ylabel)

        ###############################################################################
        # Quantitative evaluation of the model quality on the test set

    plt.show()
    
def autolabel(rects,ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')       

def plot_bar(model_list,result_mat,title,param_list,loc,log,ylabel):
    color_list=['r','b','g']
    N = len(param_list)
    
    legd=()
    label_tup=()
   
    ind_list = np.arange(N)  # the x locations for the groups
    width = 0.15       # the width of the bars
    fig, ax = plt.subplots()
    ind=0    
     
    for i in range(len(model_list)):
        if log==0:
            rect = ax.bar(ind_list+(i+1)*width, result_mat[:,i], width, color=color_list[i])
           
        else:
            rect = ax.bar(ind_list+(i+1)*width, -np.log10(result_mat[:,i]), width, color=color_list[i])

        legd=legd+(rect,)
        
    ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.set_xticks(ind_list+(i+1)*width)
    ax.set_xticklabels(param_list)

    ax.legend(legd, model_list,loc=loc)


if __name__ == '__main__':

    main()
    
