from MachineLearn.Classes.Extractors.LBP import Lbp4Bits,Lbp8Bits
import numpy as np
import cv2
import thread


def generate_files(file_path, function_class):
    new_file = []
    classes = [[1], [1], [1, 2, 3, 4], [1, 2, 3, 4]]
    for c, i in enumerate(classes):
        for p in i:
            cont = 1
            img = cv2.imread("../DataSet-Baumann/IMG_CORTADAS_ELIAS/c{}_p{}_{:05d}.JPG".format(c + 1, p, cont), 0)
            while True:
                print c + 1, p, cont
                if img is None and c > 1:
                    break
                elif cont > 400:
                    break
                elif img is not None:
                    oLbp = function_class(img)
                    oLbp.calculate_attributes()
                    new_file.append(oLbp.export_to_classifier("c{}_p{}".format(c + 1, p)))
                cont += 1
                img = cv2.imread("../DataSet-Baumann/IMG_CORTADAS_ELIAS/c{}_p{}_{:05d}.JPG".format(c + 1, p, cont), 0)
                np.savetxt(file_path, new_file, delimiter=",", fmt="%s")


t1 = thread.start_new_thread(generate_files, ("LBP_FILES/M1_4b_RE_ELIAS.txt", Lbp4Bits))
t2 = thread.start_new_thread(generate_files, ("LBP_FILES/M1_8b_RE_ELIAS.txt", Lbp8Bits))
while 1:
    pass