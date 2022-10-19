import argparse
import torch
from torchvision import datasets, transforms

import os
import numpy as np
import pandas as pd
import tqdm

from model import ConvVAE
import OneClassMnist
from gradcam import GradCAM
#import cv2
#from PIL import Image
from torchvision.utils import save_image, make_grid
from arrhythmia_loader import FukudaECGDataset
import matplotlib.pyplot as plt
import pickle

from train_expVAE import iwae

cuda = torch.cuda.is_available()
if cuda:
    print('cuda available')

device = torch.device("cuda" if cuda else "cpu")

### Save attention maps  ###
# def save_cam(image, filename, gcam):
#     gcam = gcam - np.min(gcam)
#     gcam = gcam / np.max(gcam)
#     h, w, d = image.shape
#     gcam = cv2.resize(gcam, (w, h))
#     gcam = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
#     gcam = np.asarray(gcam, dtype=np.float) + \
#         np.asarray(image, dtype=np.float)
#     gcam = 255 * gcam / np.max(gcam)
#     gcam = np.uint8(gcam)
#     cv2.imwrite(filename, gcam)

def read_file(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data

def evaluate_mll(x, model):
    num_samples = 256
    with torch.no_grad():
        mu, logvar    = model(x, op="encode")
        z, eps        = model(mu, logvar, op="reparam_eval", n_samples=num_samples)
        x_recon       = model(z, op="decode")
        mll           = iwae(0, x, x_recon, eps, z, mu, logvar)
    return mll

def main():
    parser = argparse.ArgumentParser(description='Explainable VAE MNIST Example')
    parser.add_argument('--result_dir', type=str, default='test_results', metavar='DIR',
                        help='output directory')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # model options
    parser.add_argument('--latent_size', type=int, default=8, metavar='N',
                        help='latent vector size of encoder')
    parser.add_argument('--model_path', type=str, default='./ckpt/checkpoint.pth', metavar='DIR',
                        help='pretrained model directory')
    parser.add_argument('--one_class', type=int, default=8, metavar='N',
                        help='outlier digit for one-class VAE testing')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    test_loader = torch.utils.data.DataLoader(
        FukudaECGDataset('test'),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    model = ConvVAE(args.latent_size).to(device)
    print(model)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    mu_avg, logvar_avg = 0, 1
    gcam = GradCAM(model, target_layer='encoder.19.conv2', cuda=True) 
    test_index=0

    plt.rcParams.update({
        #"figure.facecolor":  (0.0, 0.0, 1.0, 0.2),
        "axes.facecolor":    (0.0, 0.0, 1.0, 0.1),
        #"savefig.facecolor": (0.0, 0.0, 1.0, 0.2),
    })

    #df     = read_file('../../all_lead1_test_processed.pickle');
    #df     = df.reset_index(drop=True)
    #labels = pd.read_csv("../../test_labels.csv")

    model.eval()

    df        = read_file('../../all_lead1_test.gzip');
    df["mll"] = np.nan
    pbar      = tqdm.tqdm(df.iterrows(), total=len(df))
    for (i, row) in pbar:
        signal = row["lead1"].astype(np.float32)
        signal = signal[0:5000]
        mu     = np.mean(signal)
        sigma  = np.std(signal)
        signal = (signal - mu) / (sigma + 1e-10)
        x      = torch.tensor(signal)
        x      = torch.unsqueeze(x, 0)
        x      = torch.unsqueeze(x, 0)
        x      = x.to(device)
        mll    = evaluate_mll(x, model)
        df["mll"][i] = mll
        pbar.set_description("mll = {:6.1f}".format(mll))
        #continue

        x      = x.repeat(2, 1, 1)
        #print(x.size())
        x_rec, mu, logvar = gcam.forward(x)

        model.zero_grad()
        gcam.backward(mu, logvar, mu_avg, logvar_avg)
        gcam_map = gcam.generate() 

        #fig, ax1 = plt.subplots()
        #ax2 = ax1.twinx()

        plt.figure(figsize=(10,1.5), tight_layout=True)
        y = x[0,:,:].squeeze().cpu().data.numpy()
        g = gcam_map[0].squeeze().cpu().data.numpy()
        extent = [0, len(y), np.min(y), np.max(y)]

        g     = np.exp(g+1)
        vmax  = np.max(g)
        alpha = np.clip(g / vmax, 0, 1.0)
        alpha = alpha[np.newaxis, :]
        plt.plot(y, "black", linewidth=1)
        plt.xlabel("t (seconds)")
        g     = g[np.newaxis, :]
        plt.imshow(g, cmap="Reds", extent=extent, aspect="auto", alpha=alpha)#, vmin=0, vmax=vmax)

        frame = plt.gca()
        frame.axes.yaxis.set_visible(False)
        frame.axes.xaxis.set_ticklabels([0, 2, 4, 6, 8, 10])
        plt.show()
        #plt.savefig("test_gradcam/{}.svg".format(i))
    df.to_pickle("all_lead1_test.gzip")

if __name__ == '__main__':
    main()
