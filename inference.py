from mivolo.predictor import Predictor
import cv2
import torch
from types import SimpleNamespace

### Camera Setup

camera_module_path =""

cap = cv2.VideoCapture(camera_module_path)

# Configuration dictionary with new keys
config_dict = {
    'output': 'output',
    'detector_weights': 'weights/yolov8x_person_face.pt',
    'checkpoint': 'weights/model_imdb_age_gender_4.22.pth.tar',
    'device': 'cuda:0',
    'with_persons': True,
    'draw': True ,
    'disable_faces' : False
} 

config = SimpleNamespace(**config_dict)

if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

model = Predictor(config = config , verbose= True)

while cap.isOpened():
    ret, frame = cap.read()
    
    # If a frame was returned, save it and display it
    if ret:
        # Display the frame in a window 
        
        faces , out_iamgesd = model.recognize(frame)
        cv2.imshow("results" , out_iamgesd)
        # Wait for 1 ms and check if the user presses the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
