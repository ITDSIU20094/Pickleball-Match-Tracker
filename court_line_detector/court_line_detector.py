import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Khởi tạo model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 28)  # 12 keypoints * 2
        
        state_dict = torch.load(model_path, map_location='cpu')
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
        keypoints = outputs.squeeze().cpu().numpy()  
        
        keypoints[0::2] = keypoints[0::2] * (w / 224.0)
        keypoints[1::2] = keypoints[1::2] * (h / 224.0)
        return keypoints

    # def draw_keypoints(self, image, keypoints, color=(0, 255, 0), radius=4):
    #     for i in range(0, len(keypoints), 2):
    #         x, y = int(keypoints[i]), int(keypoints[i+1])
    #         if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
    #             cv2.circle(image, (x, y), radius, color, -1)
    #             cv2.putText(image, str(i//2), (x+5, y-5), 
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    #     return image
    
    ## Update draw_keypoints to only draw the first 12 keypoints (6 pairs)
    def draw_keypoints(self, image, keypoints, color=(0, 255, 0), radius=4, num_keypoints=12):
        max_coords = num_keypoints * 2
        for i in range(0, min(len(keypoints), max_coords), 2):
            x, y = int(keypoints[i]), int(keypoints[i+1])
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(image, (x, y), radius, color, -1)
                cv2.putText(image, str(i//2), (x+5, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
    
    @staticmethod
    def get_court_corners(keypoints):
        """
        keypoints: numpy array shape (28,) - 14 points (x0,y0,...,x13,y13)
        Returns: tuple of indices (tl, tr, br, bl) of the four corners
        """
        xs = [keypoints[i] for i in range(0, 28, 2)]
        ys = [keypoints[i+1] for i in range(0, 28, 2)]
        # Top-left: min(x+y)
        tl = np.argmin([xs[i] + ys[i] for i in range(14)])
        # Top-right: max(x-y)
        tr = np.argmax([xs[i] - ys[i] for i in range(14)])
        # Bottom-right: max(x+y)
        br = np.argmax([xs[i] + ys[i] for i in range(14)])
        # Bottom-left: min(x-y)
        bl = np.argmin([xs[i] - ys[i] for i in range(14)])
        return tl, tr, br, bl