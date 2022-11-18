import cv2
import tensorflow.keras
import numpy as np
import time

np.set_printoptions(suppress=True)
model=tensorflow.keras.models.load_model("keras_model.h5")

with open("labels.txt" , "r") as file:
    class_names=file.read().split('\n')

data=np.ndarray(shape=(1,224,224,3),dtype=np.float32)
size=(224,224)
cap=cv2.VideoCapture(0)
count_matured=0
count_nonmatured=0
while cap.isOpened():
    start=time.time()
    ret,img=cap.read()
    height,width,channel=img.shape
    scale_value=width/height
    img_resized=cv2.resize(img,size,fx=scale_value,fy=1,interpolation=cv2.INTER_NEAREST)
    img_array=np.asarray(img_resized)
    normalized=(img_array.astype(np.float32)/127.0)-1
    data[0]=normalized
    prediction=model.predict(data)

    index=np.argmax(prediction)
    class_name=class_names[index]
    confidence_score=prediction[0][index]

    end=time.time()
    run_time=start-end

    fps=1/run_time
    if class_name=="matured":
        time.sleep(2)
        count_matured+=1
    elif class_name=="tender":
        time.sleep(2)
        count_nonmatured+=1
    cv2.putText(img,class_name,(75,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
    cv2.imshow("Monitoring",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"Matured coconut = {count_matured}")
        print(f"Non Matured cocnut = {count_nonmatured}")
        print(f"Total coconut = {count_matured+count_nonmatured}")
        break
cv2.destroyAllWindows()
cap.release()
