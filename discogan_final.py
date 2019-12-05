import argparse
import os
import torch
import torchvision
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from PIL import Image
from network import Generator

parser = argparse.ArgumentParser(description='DiscoGAN in One Code')

# Hyper-parameters
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')

# misc
parser.add_argument('--model_path', type=str, default='C:/Capstone/ai_dataset/discogan_models/')  # Model Tmp Save
parser.add_argument('--load_epoch', type=int, default=-1)
parser.add_argument('--data_root', type=str, default='C:/Capstone/server_dataset/')

# AutoDrawer
parser.add_argument('--token', type=str, default='asdf123', help='your token')
parser.add_argument('--cagetory', type=str, default='handbag')


##### Helper Functions for Data Loading & Pre-processing
class ImageFolder(data.Dataset):
    def __init__(self, opt):
        self.transformS = transforms.Compose([transforms.Scale((64, 64)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5),
                                                                   (0.5, 0.5, 0.5))])
        self.image_len = None
        self.dir = args.data_root

        input_dir = os.path.join(self.dir, args.token + '/', 'TextureGAN/')
        filename = '%s.jpg' % args.category
        self.out_category = 'shoes' if args.category == 'handbag' else 'handbag'
        imgpath = os.path.join(input_dir, filename)

        self.image_paths_A = [imgpath]
        # self.image_paths_A = list(map(lambda x: os.path.join(self.dir, x), os.listdir(self.dir)))
        self.image_len = len(self.image_paths_A)

    def __getitem__(self, index):
        A_path = self.image_paths_A[index]
        A = self.transformS(Image.open(A_path).convert('RGB'))

        return A

    def __len__(self):
        return self.image_len


##### Helper Function for GPU Training
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


##### Helper Function for Math
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


######################### Main Function
def main():
    # Pre-settings
    cudnn.benchmark = True
    global args
    args = parser.parse_args()
    print(args)

    dataset = ImageFolder(args)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=args.batchSize,
                                  shuffle=True,
                                  num_workers=2)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

    # Load model
    if args.load_epoch != 0:
        epoch = args.load_epoch

        if epoch == -1:
            with open(os.path.join(args.model_path, 'checkpoint.txt')) as f:
                epoch = int(f.readline())

        print('Epoch %d has loaded.' % epoch)
    else:
        epoch = 0

    # Networks
    g_pathAtoB = os.path.join(args.model_path, 'generatorAtoB-%d.pkl' % epoch)

    generator_AtoB = Generator()
    generator_AtoB.load_state_dict(torch.load(g_pathAtoB))
    generator_AtoB.eval()

    if torch.cuda.is_available():
        generator_AtoB = generator_AtoB.cuda()

    """Train generator and discriminator."""
    for i, sample in enumerate(data_loader):
        A = to_variable(sample)
        A_to_B = generator_AtoB(A)

        # save the sampled images
        # res = torch.cat((A, A_to_B), dim=2)
        res = A_to_B
        out_dir = os.path.join(args.data_root, args.token + '/', 'discoGAN/')
        out_filename = os.path.join(out_dir, '%s.jpg' % dataset.out_category)
        torchvision.utils.save_image(denorm(res.data), os.path.join(out_dir, out_filename))


if __name__ == "__main__":
    main()
