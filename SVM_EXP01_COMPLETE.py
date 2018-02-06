from MachineLearn.Classes.experiment import Experiment
from MachineLearn.Classes.data_set import DataSet
from MachineLearn.Classes.data import Data
import numpy as np
import cv2

oExp = Experiment()
basemask = np.array([1,2,5,9,15,16,17,21,22,23,25])
svmVectors = []
samples_per_class = 60

###############################################################################################################################
basemask = basemask-1
for i in range(1):
        oDataSet = DataSet()
        base = np.loadtxt("GLCM_FILES/M1_CM8b.txt", usecols=(x for x in range(24)), delimiter=" ") 
        classes  = np.loadtxt("GLCM_FILES/M1_CM8b.txt",dtype=object, usecols=(24), delimiter=" ") 
        classes = np.matrix(classes)
        base = np.array(np.hstack((base,classes.T)))
        base[base == "c1_p1"] = "c1"
        base[base == "c2_p1"] = "c2"        
        base[base == "c3_p1"] = "c3"
        base[base == "c3_p2"] = "c3"
        base[base == "c3_p3"] = "c3"
        base[base == "c3_p4"] = "c3"
        base[base == "c4_p1"] = "c4"
        base[base == "c4_p2"] = "c4"
        base[base == "c4_p3"] = "c4"
        base[base == "c4_p4"] = "c4"
        unique, counts = np.unique(base[:,-1], return_counts=True)
        for c in ["c1","c2","c3","c4"]:
                to_add = base[np.nonzero(base==c)[0]]
                if to_add.shape[0] > samples_per_class:
                        to_add = np.nonzero(base==c)[0]
                        np.random.shuffle(to_add)
                        to_add = base[to_add[:samples_per_class]]
                for k in to_add:
                        oDataSet.addSampleOfAtt(k[basemask])
        oDataSet.normalizeDataSet()  
        for j in range(50):
                oData  = Data(4, 20, samples=60)
                oData.randomTrainingTestPerClass()
                svm = cv2.SVM()
                oData.params = dict(kernel_type = cv2.SVM_RBF,svm_type = cv2.SVM_C_SVC,gamma=2.0,nu = 0.0,p = 0.0, coef0 = 0)
                svm.train(np.float32( oDataSet.atributes[oData.Training_indexes]) , np.float32( oDataSet.labels[oData.Training_indexes]) , params = oData.params)
                svmVectors.append(svm.get_support_vector_count())
                results = svm.predict_all(np.float32(oDataSet.atributes[oData.Testing_indexes])) 
                oData.setResultsFromClassfier(results, oDataSet.labels[oData.Testing_indexes])
                oDataSet.append(oData)
        oExp.addDataSet(oDataSet, description="  50 execucoes M=1 CM=8b base BAUMMAN ")
oExp.save("OBJETOS/EXP_01_ACC_M1_50_CM8b_BAUMMAN.txt")
print oExp
print min(svmVectors), max(svmVectors), np.average(svmVectors) , np.std(svmVectors)

###############################################################################################################################