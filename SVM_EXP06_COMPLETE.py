from MachineLearn.Classes.experiment import Experiment
import numpy as np

oExp = Experiment()
svmVectors = []
qtd_per_class = np.array([175, 200])

# #################################################################################################################################
# from MachineLearn.Classes.data_set import DataSet
# from MachineLearn.Classes.data import Data
# import cv2

# for i in range(1):
#     oDataSet = DataSet()
#     base = np.loadtxt("LBP_FILES/M1_4b_RE_ELIAS.txt", usecols=(x for x in range(16)), delimiter=",")
#     classes = np.loadtxt("LBP_FILES/M1_4b_RE_ELIAS.txt", dtype=object, usecols=16, delimiter=",")
#     classes = np.matrix(classes)
#     base = np.array(np.hstack((base, classes.T)))
#     base[base == "c1_p1"] = "c1 e c2"
#     base[base == "c2_p1"] = "c1 e c2"
#     base[base == "c3_p1"] = "c3 e c4"
#     base[base == "c3_p2"] = "c3 e c4"
#     base[base == "c3_p3"] = "c3 e c4"
#     base[base == "c3_p4"] = "c3 e c4"
#     base[base == "c4_p1"] = "c3 e c4"
#     base[base == "c4_p2"] = "c3 e c4"
#     base[base == "c4_p3"] = "c3 e c4"
#     base[base == "c4_p4"] = "c3 e c4"
#
#     unique, counts = np.unique(base[:, -1], return_counts=True)
#     for c in ["c1 e c2", "c3 e c4"]:
#         to_add = base[np.nonzero(base == c)[0]]
#         for k in to_add:
#             oDataSet.addSampleOfAtt(k)
#     oDataSet.normalizeDataSet()
#     for j in range(50):
#         oData = Data(2, 20, samples=60)
#         oData.randomTrainingTestByPercent(qtd_per_class.copy(), 0.80)
#         svm = cv2.SVM()
#         oData.params = dict(kernel_type=cv2.SVM_RBF, svm_type=cv2.SVM_C_SVC, gamma=2.0, nu=0.0, p=0.0, coef0=0,
#                             k_fold=2)
#         svm.train_auto(np.float32(oDataSet.atributes[oData.Training_indexes]),
#                        np.float32(oDataSet.labels[oData.Training_indexes]), None, None, params=oData.params)
#         svmVectors.append(svm.get_support_vector_count())
#         results = svm.predict_all(np.float32(oDataSet.atributes[oData.Testing_indexes]))
#         oData.setResultsFromClassfier(results, oDataSet.labels[oData.Testing_indexes])
#         oDataSet.append(oData)
#         print j
#     oExp.addDataSet(oDataSet, description="  50 execucoes M=1 LBP=4b base BAUMMAN ")
#     print oDataSet
#     print
# oExp.save("OBJETOS/EXP_06_ACC_M1_50_LBP4b_BAUMMAN.txt")
# print oExp
# print min(svmVectors), max(svmVectors), np.average(svmVectors), np.std(svmVectors)
#
# ##############################################################################################################################

oExp = oExp.load("OBJETOS/EXP_06_ACC_M1_50_LBP4b_BAUMMAN.txt")
print oExp
print oExp.experimentResults[0].sum_confusion_matrix / 50
