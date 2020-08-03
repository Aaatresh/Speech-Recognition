## Importing necessary libraries

from __future__ import print_function
import warnings
import os
from scikits.talkbox.features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np
warnings.filterwarnings('ignore')

#___________________________________________________________________________________________________________________

def extract_mfcc(full_audio_path):
    
    """
        Method to extract MFCC from a given audio file path
        
        Arguments:
            full_audio_path: path of audio file from which MFCC is to be extracted.
        
        Returns:
            mfcc_features: the mel fequency cepstral coefficients extracted from the audio file.
    """
    
    sample_rate, wave =  wavfile.read(full_audio_path)
    mfcc_features = mfcc(wave, nwin=int(sample_rate * 0.03), fs=sample_rate, nceps=12)[0]
    return mfcc_features

def buildDataSet(root):

    """
        Method to build the dataset given a number of audio files. These are read given the root directory.
        
        Arguements:
            root: root directory of files.
        
        Returns:
            dataset: a dictionary of audio files. These are pairs of mfcc vectors and their corresponding label.
    """
    
    dirlist = os.listdir(root)
    dataset = {}
    for dir in dirlist:
    
        label = dir
        dir = os.path.join(root,dir)
    
        for files in os.listdir(dir):
         
            feature = extract_mfcc(os.path.join(dir,files))
            
            if label not in dataset.keys():
                dataset[label] = []
                dataset[label].append(feature)
            else:
                exist_feature = dataset[label]
                exist_feature.append(feature)
                dataset[label] = exist_feature
                
    return dataset

def train_GMMHMM(dataset):
    
    """
        Method to grain the gaussian mixture model-hidden markov model. 
        
        Arguments:
            dataset: a dictionary of audio files paired with their corresponding labels.
            
        Returns:
            a dictionary of gaussian mixture model-hidden markov models for every label
    """
    
    GMMHMM_Models = {}
    
    states_num = 5
    GMM_mix_num = 3
    tmp_p = 1.0/(states_num-2)
    transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], \
                               [0, tmp_p, tmp_p, tmp_p , 0], \
                               [0, 0, tmp_p, tmp_p,tmp_p], \
                               [0, 0, 0, 0.5, 0.5], \
                               [0, 0, 0, 0, 1]],dtype=np.float)


    startprobPrior = np.array([0.5, 0.5, 0, 0, 0],dtype=np.float)

    for label in dataset.keys():
       
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
                           transmat_prior=transmatPrior, startprob_prior=startprobPrior, \
                           covariance_type='diag', n_iter=10)
        
        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
        
        trainData = np.vstack(trainData)
        model.fit(trainData, lengths=length)  # get optimal parameters
        
        GMMHMM_Models[label] = model
    
    return GMMHMM_Models

def main():
    """
        Method to grain the gaussian hidden markov model. 
        
        Arguments:
            dataset: a dictionary of audio files paired with their corresponding labels.
            
        Returns:
            a dictionary of gaussian hidden markov models for every label
    """
    
    trainDir = './train_audio/'
    trainDataSet = buildDataSet(trainDir)
    
    hmmModels = train_GMMHMM(trainDataSet)

    testDir = './test_audio/'
    testDataSet = buildDataSet(testDir)

    score_cnt = 0
    total_cnt = 0
    
    for label in testDataSet.keys():
        feature = testDataSet[label]
        
        for i in range(len(feature)):
        
            scoreList = {}
            
            for model_label in hmmModels.keys():
                model = hmmModels[model_label]
                score = model.score(feature[i])
                scoreList[model_label] = score
            
            predict = max(scoreList, key=scoreList.get)
            
            if predict == label:
                score_cnt+=1
            
            total_cnt += 1
            
    print("Final recognition rate is %.2f"%(100.0*score_cnt/total_cnt), "%")

#___________________________________________________________________________________________________________________

if __name__ == '__main__':
    main()
