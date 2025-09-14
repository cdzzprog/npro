import torch
from einops import rearrange
from torchvision import transforms
from PIL import Image

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


if __name__ == '__main__':

    image_path = r'E:\dataset1\\bijie\landlside\train\opt\df002.png'
    image = Image.open(image_path)
    transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),            # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image=torch.unsqueeze(image, 0)
    print(image.size())
    
    input = torch.randn(1, 3, 256, 256)  # 假设输入tensor B C H W
    output = to_3d(image)
    print(output.size())    #输出shape b n c

    output1 =to_4d(output, 256, 256)  # 指定高宽 h*w =n
    print(output1.size())