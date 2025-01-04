import torch
import torch.nn as nn
from PIL import Image
import os
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import customtkinter as ctk  # 这里假设你使用了 customtkinter 库
from tools import normalize
from ResUnet import ResUnet
from Unet import Unet
from EGEUNet import EGEUNet

class EvaluationApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Model Evaluation")
        self.geometry("500x400")

        # Initialize button to trigger evaluation
        self.eval_button = ctk.CTkButton(self, text="Start Evaluation", command=self.run_evaluation)
        self.eval_button.pack(pady=20)

    def run_evaluation(self):
        test_path1 = r"E:\papers\model\GVLM_CD\dataset1\GVLM-CD256\test/im1/"
        test_path2 = r"E:\papers\model\GVLM_CD\dataset1\GVLM-CD256\test/im2/"
        label_path = r"E:\papers\model\GVLM_CD\dataset1\GVLM-CD256\test/label/"

        # Initialize the model
        model = EGEUNet()
        device = torch.device('cuda:0')
        model.to(device)
        model.load_state_dict(torch.load("E:\papers\model\GVLM_CD\GVLM-main\snapshot/2024-12-19_21_42_23_EGEunet_50.pth"))
        model.eval()

        TP, FN, FP, TN = 0, 0, 0, 0

        # Create result directory if it does not exist
        result_path = './result2/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # Loop through all images in the test dataset
        for i in tqdm(range(len(os.listdir(test_path1)))):
            # Load input images (RGB images)
            img1 = Image.open(test_path1 + str(i) + ".jpg")
            img2 = Image.open(test_path2 + str(i) + ".jpg")
            
            # Normalize and prepare the input tensors
            img1 = normalize(np.array(img1))
            img2 = normalize(np.array(img2))
            
            split1 = torch.from_numpy(img1.transpose((2, 0, 1)))  # Convert to CxHxW format
            split2 = torch.from_numpy(img2.transpose((2, 0, 1)))  # Convert to CxHxW format

            split1 = Variable(torch.unsqueeze(split1, dim=0).float(), requires_grad=False)
            split2 = Variable(torch.unsqueeze(split2, dim=0).float(), requires_grad=False)
            
            split1 = split1.to(device)
            split2 = split2.to(device)
            
            # Perform prediction using the model
            pred = model(split1, split2)
            if len(pred) == 2:
                pred = pred[1]
            # Threshold the predictions (0 or 1)
            zero = torch.zeros_like(pred)
            one = torch.ones_like(pred)
            pred = torch.where(pred > 0.7, one, pred)
            pred = torch.where(pred <= 0.7, zero, pred)
            
            # Convert prediction to numpy array and squeeze unnecessary dimensions
            pred = pred.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0))
            
            # Convert the RGB images back to uint8
            img1_rgb = (img1 * 255).astype(np.uint8) if img1.max() <= 1 else img1.astype(np.uint8)
            img2_rgb = (img2 * 255).astype(np.uint8) if img2.max() <= 1 else img2.astype(np.uint8)
            
            # Load the ground truth label for evaluation
            ref = Image.open(label_path + str(i) + ".jpg")
            label = np.array(ref).astype(np.uint8)
            
            # Plot images: im1, im2, label, and prediction in the same column
            fig, axes = plt.subplots(1, 4, figsize=(15, 5))  # 1 row, 4 columns
            
            axes[0].imshow(img1_rgb)
            axes[0].set_title('Image 1 (RGB)')
            axes[0].axis('off')
            
            axes[1].imshow(img2_rgb)
            axes[1].set_title('Image 2 (RGB)')
            axes[1].axis('off')
            
            axes[2].imshow(label, cmap='gray')  # Use gray colormap for the label
            axes[2].set_title('Label')
            axes[2].axis('off')
            
            axes[3].imshow(pred, cmap='gray')  # Use gray colormap for prediction
            axes[3].set_title('Prediction')
            axes[3].axis('off')
            
            # Save the combined plot as an image
            plt.tight_layout()
            plt.savefig(os.path.join(result_path, f"{i}_result.png"))
            plt.close()

            # Calculate TP, FN, FP, TN
            res = np.squeeze(pred, -1).astype(np.int64)
            label[label == 255] = 1  # Normalize label (255 -> 1)
            
            TP += np.sum(res * label == 1)
            FN += np.sum(label * (1 - res) == 1)
            FP += np.sum(res * (1 - label) == 1)
            TN += np.sum((1 - res) * (1 - label) == 1)

        # Print evaluation results
        print(f'TP={TP} | TN={TN} | FP={FP} | FN={FN}')

        # Calculate metrics
        Accu = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        Sensitivity = TP / (TP + FN)
        F1 = 2 * ((Precision * Recall) / (Precision + Recall))

        pe = ((TP + FN) * (TP + FP) + (TN + FP) * (TN + FN)) / ((TP + TN + FP + FN) ** 2)
        kappa = (Accu - pe) / (1 - pe)
        IoU = TP / (TP + FP + FN)

        print(f'Accu={Accu} Precision={Precision} Recall={Recall} F1={F1} kappa={kappa} IoU={IoU} Specificity={Specificity} Sensitivity={Sensitivity}')

if __name__ == "__main__":
    app = EvaluationApp()
    app.mainloop()
