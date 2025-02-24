import cv2
import glob

pic = glob.glob(r"F:\deeplearning\pytorch\Reinforcement_Learning\one_e\template\*.png")

for i in pic:
    image = cv2.imread(i)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cv2.imwrite(i,image)