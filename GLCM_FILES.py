from MachineLearn.Classes.Extractors.GLCM import GLCM
import numpy as np 
import cv2
new_file = []
classes = [[1],[1],[1,2,3,4],[1,2,3,4]]
for c,i in enumerate(classes):
        for p in i:
                cont = 1
                img = cv2.imread("../DataSet-Baumann/IMAGENS_RECORTADAS/c{}_p{}_{:05d}.JPG".format(c+1,p,cont), 0)
                while(img!=None):
                        print c+1, p,cont                      
                        oGLCM = GLCM(img, 8)
                        oGLCM.generateCoOccurenceHorizontal()
                        oGLCM.normalizeCoOccurence()
                        oGLCM.calculateAttributes()
                        new_file.append(oGLCM.exportToClassfier((c+1)*10+p))
                        cont+=1
                        img = cv2.imread("../DataSet-Baumann/IMAGENS_RECORTADAS/c{}_p{}_{:05d}.JPG".format(c+1,p,cont), 0)                            
                        np.savetxt("03.txt",new_file)