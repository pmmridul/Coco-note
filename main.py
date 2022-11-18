from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np
import time

matured=0
nonmatured=0

def load_labels(path): # Read the labels from the text file as a Python list.
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
  set_input_tensor(interpreter, image)

  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)

  ordered = np.argpartition(-output, 1)
  return [(i, output[i]) for i in ordered[:top_k]][0]

data_folder = "/home/pi/Downloads/coconut_tenserflow/"
model_path = data_folder + "mobilenet_v1_1.0_224_quant.tflite"
label_path = data_folder + "labels_mobilenet_quant_v1_224.txt"

interpreter = Interpreter(model_path)
print("Model Loaded Successfully.")

interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
# print("Image Shape (", width, ",", height, ")")

# Load an image to be classified.
cap=cv2.VideoCapture(0)
while True:
    ret,Image2=cap.read()
    Image1=cv2.resize(Image2,(width,height))
    Image=cv2.cvtColor(Image1,cv2.COLOR_BGR2RGB)

# Classify the image.
    time1 = time.time()
    label_id, prob = classify_image(interpreter, Image)
    time2 = time.time()
    classification_time = np.round(time2-time1, 3)
    print("Classificaiton Time =", classification_time, "seconds.")

    # Read class labels.
    labels = load_labels(label_path)

    # Return the classification label of the image.
    classification_label = labels[label_id]
    cv2.putText(Image2, classification_label, (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.imshow("Monitoring", Image2)

    if classification_label=="0 matured":
        matured+=1
    elif classification_label=="1 tender":
        nonmatured+=1
    else:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(f"Total Count :{nonmatured+matured} Non-matured:{nonmatured} Matured:{matured}")
cv2.destroyAllWindows()
cap.release()
print("Image Label is :", classification_label, ", with Accuracy :", np.round(prob*100, 2), "%.")