from MachineLearn.Classes.Extractors.GLCM import GLCM
import numpy as np 
import cv2
new_file = []
classes = [[],[1],[1,2,3,4],[1,2,3,4]]
for c,i in enumerate(classes):
        for p in i:
                cont = 1
                img = cv2.imread("../DataSet-Baumann/IMG_CORTADAS_ELIAS/c{}_p{}_{:05d}.JPG".format(c+1,p,cont), 0)
                while(True):
                        print c+1, p,cont                                              
                        if (img==None and c>1):break
                        elif (cont>400): break
                        elif(img!=None):
                                oGLCM = GLCM(img, 8)
                                oGLCM.generateCoOccurenceHorizontal()
                                oGLCM.normalizeCoOccurence()
                                oGLCM.calculateAttributes()
                                new_file.append(oGLCM.exportToClassfier("c{}_p{}".format(c+1,p)))
                        
                        cont+=1
                        img = cv2.imread("../DataSet-Baumann/IMG_CORTADAS_ELIAS/c{}_p{}_{:05d}.JPG".format(c+1,p,cont), 0)                            
                        np.savetxt("GLCM_FILES/M1_CM8b_RE_ELIAS2.txt",new_file,delimiter=",", fmt="%s")