
import roslibpy
import base64
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from BalloonNetCNNBOX import BalloonNetCNN
from PIL import Image as PILImage

# Define the inference transforms (matching the training pipeline)
transform = transforms.Compose([
    transforms.Resize((832, 368)),  # Resize the image
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize (ImageNet stats)
])
#after convolution sizes
model_width = 832/2/2/2
model_height = 368/2/2/2
class imgSubscriber():
    """creates an image subscriber, inherit from node. uses cv bridge to convert"""
    def __init__(self):
        self.model = BalloonNetCNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model.load_state_dict(torch.load("balloon_pos.pth", map_location=self.device))
        except Exception as e:
            print("load failed")
            print(e)
        #model loaded, set to eval
        self.model.eval()
        self.model.to(self.device)
        
        self.ros = roslibpy.Ros(host = 'localhost', port=9090)
        self.ros.run()
        # triggers self.listener, takes in topic /image in Image message format
        self.subscription = roslibpy.Topic(self.ros, '/Unity/Image', 'std_msgs/String')
        self.publisher = roslibpy.Topic(self.ros, "/Ros/Coords", "std_msgs/Float32MultiArray")
        self.subscription.subscribe(self.listener)
    
    def sender(self, image, width, height):
        try:
            transformed_image = transform(image).unsqueeze(0)
            input_image = transformed_image.to(self.device)
            width_scale = width /model_width
            height_scale = height/ model_height
            scale = torch.tensor([width_scale,height_scale,width_scale,height_scale]).to(self.device)
            with torch.no_grad():
                output = self.model(input_image) *scale
                
                print(output[0][0].item())
                print(output[0][1].item())
                print(output[0][2].item())
                print(output[0][3].item())
            self.publisher.publish(roslibpy.Message({'data':[output[0][0].item(),
                                                      output[0][1].item(),
                                                      output[0][2].item(),
                                                      output[0][3].item()]}))
        except:
            print("failed")
        print("coords sent")
        
        
    def listener(self, msg):
        print("image received")
        try:
            # Step 1: Decode image from base64 (rosbridge format)
            img_bytes = base64.b64decode(msg['data'])
            # print(img_bytes)

            # Step 2: Convert to NumPy array (like cv_bridge would)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            print(f"Decoded byte size: {len(img_bytes)}")
            with open("raw_image.png", "wb") as f:
                f.write(img_bytes)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)


            cv2.imwrite("output.png", cv_image)
            print("saved")
            pil_img = PILImage.fromarray(cv_image)
            width, height = pil_img.size

            self.sender(pil_img, width, height)
        except Exception as e:
            print(f"Failed to convert image: {e}")
    
    def spin(self):
        try:
            while self.ros.is_connected:
                pass
        except KeyboardInterrupt:
            print("cya")
            self.subscription.unsubscribe()
            self.publisher.unsubscribe()
            self.ros.terminate()
        
        

def main(args = None):
    print("starting")
    node = imgSubscriber()
    node.spin()

if __name__ == '__main__':
    main()