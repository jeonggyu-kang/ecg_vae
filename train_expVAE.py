
import argparse
import torch
import torch.optim as optim
from torch.nn            import functional as F
from torchvision         import datasets, transforms
from torchvision.utils   import save_image, make_grid
from torch.distributions import Normal

import pickle
import os
import shutil
import numpy as np
import tqdm
import pandas
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from arrhythmia_loader import FukudaECGDataset
from model             import ConvVAE

cuda = torch.cuda.is_available()
if cuda:
    print('cuda available')

device = torch.device("cuda" if cuda else "cpu")
host   = torch.device("cpu")

def loss_function(recon_x, x, mu, logvar):
    batch_size = x.size()[0]
    likelihood = F.mse_loss(recon_x, x, reduction='sum') / batch_size
    kld        = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return likelihood, kld

### Training #####
def train(epoch, model, train_loader, optimizer, args):
    model.train()
    train_loss = 0

    pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (data, _) in pbar:
        data = data.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        like, kld   = loss_function(recon_batch, data, mu, logvar)
        loss        = like + args.beta*kld
        train_loss += loss.item() / len(train_loader)

        loss.backward()
        optimizer.step()

        pbar.set_description("loss = {:8.1f}, kl = {:8.1f}, recon error = {:8.1f}".format(
            loss.item(), kld.item(), like.item()))
    return train_loss

def iwae(epoch, x, x_recon, eps, z, mu, logvar):
    #n_samples = int(np.sqrt(z.size()[0]))
    n_samples = z.size()[0]

    # Calculating P(x,h)
    log_p_z   = torch.sum(-0.5*z**2
                          - 0.5*torch.log(2*z.new_tensor(np.pi)), -1)
    loglike   = torch.sum(-0.5*(x - x_recon)**2, [1,2])
    log_p_x_z = loglike + log_p_z 
    log_q_z   = torch.sum(-0.5*(eps)**2
                          - 0.5*torch.log(2*z.new_tensor(np.pi))
                          - 0.5*logvar,
                          -1)

    log_w     = (log_p_x_z - log_q_z)
    log_w_mat = log_w#log_w.view(n_samples, n_samples)
    log_w_sum = torch.logsumexp(log_w_mat, -1) - np.log(n_samples)
    #return torch.mean(log_w_sum).item()
    return log_w_sum.item()

def render_sample(name, x, x_recon, diagnose, loss, mll):
    fs = 500
    t  = np.linspace(0, len(x)/fs, len(x))

    plt.plot(t, x, 'b', label="ECG", linewidth=0.5)
    plt.plot(t, x_recon[1:16,:].T, 'r', # label="Recon. ECG", 
             linewidth=0.5, alpha=0.1)
    plt.title("loss = {:8.1f}, mll = {:8.1f}, diagnose = {}".format(loss, mll, diagnose))
    plt.xlabel('time (sec)')
    plt.legend(loc="lower right")
    plt.savefig(name, dpi=300)
    plt.clf()
    return

def test(epoch, model, normal_loader, arrhythm_loader, args):
    model.eval()

    render_path = "samples/{}".format(epoch)
    if not os.path.exists(render_path):
        os.makedirs(render_path)

    results = []
    with torch.no_grad():
        pbar = tqdm.tqdm(enumerate(normal_loader), total=len(normal_loader))
        for img_idx, (x, diagnose) in pbar:
            x             = x.to(device)
            mu, logvar    = model(x, op="encode")
            z, eps        = model(mu, logvar,
                                  op="reparam_eval",
                                  n_samples=args.num_samples*args.num_samples)
            x_recon       = model(z, op="decode")
            x_recon_mean  = model(mu, op="decode")
            like, kld     = loss_function(x_recon_mean, x, mu, logvar)
            loss          = (like + args.beta*kld).item()
            mll           = compute_mll(epoch, x, x_recon, eps, z, mu, logvar)

            results.append({"mll"  : mll,
                            "loss" : loss,
                            "type" : "normal",
                            "mu"   : mu.cpu().numpy(),
                            "diagnose" : diagnose})

            if img_idx < 50:
                render_sample("samples/{}/normal_{}.png".format(epoch, img_idx),
                              torch.squeeze(x).cpu().numpy(),
                              torch.squeeze(x_recon).cpu().numpy(),
                              diagnose,
                              loss, mll)
            pbar.set_description("normal loss = {:6.1f}, mll = {:6.1f}".format(loss, mll))

        pbar = tqdm.tqdm(enumerate(arrhythm_loader), total=len(arrhythm_loader))
        for img_idx, (x, diagnose) in pbar:
            x             = x.to(device)
            mu, logvar    = model(x, op="encode")
            z, eps        = model(mu, logvar,
                                  op="reparam_eval",
                                  n_samples=args.num_samples*args.num_samples)
            x_recon       = model(z, op="decode")
            x_recon_mean  = model(mu, op="decode")
            like, kld     = loss_function(x_recon_mean, x, mu, logvar)
            loss          = (like + args.beta*kld).item()
            mll           = iwae(epoch, x, x_recon, eps, z, mu, logvar)

            results.append({"mll"  : mll,
                            "loss" : loss,
                            "type" : "arrhythm",
                            "mu"   : mu.cpu().numpy(),
                            "diagnose" : diagnose})

            if img_idx < 50:
                render_sample("samples/{}/arrhythm_{}.png".format(epoch, img_idx),
                              torch.squeeze(x).cpu().numpy(),
                              torch.squeeze(x_recon).cpu().numpy(),
                              diagnose,
                              loss, mll)
            pbar.set_description("arrhythm loss = {:6.1f}, mll = {:6.1f}".format(loss, mll))

    with open("samples/{}/validation.pickle".format(epoch), "wb") as f:
        pickle.dump(results, f)

    mll_mean  = np.mean([entry["mll"]  for entry in results])
    loss_mean = np.mean([entry["loss"] for entry in results])
    return mll_mean, loss_mean

def save_checkpoint(state, is_best, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    checkpoint_file = os.path.join(outdir, 'checkpoint.pth')
    best_file = os.path.join(outdir, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)

def main():
    parser = argparse.ArgumentParser(description='Explainable VAE MNIST Example')
    parser.add_argument('--result_dir', type=str, default='train_results', metavar='DIR',
                        help='output directory')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt', metavar='DIR',
                        help='ckpt directory')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--num_samples', type=int, default=64, metavar='N',
                        help='number of samples for estimating marginal likelihood (default: 64)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--resume', default=False, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None')

    # model options
    parser.add_argument('--latent_size', type=int, default=8, metavar='N',
                        help='latent vector size of encoder')
    parser.add_argument('--beta',       type=float, default=1.0, metavar='VAL',
                        help='disentanglement strength in beta-VAE (default:1))')


    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    font_list = list(filter(lambda x : "nanum" in x, font_list))
    font_name = fm.FontProperties(fname=font_list[0]).get_name()
    plt.rcParams["font.family"] = font_name
    plt.figure(figsize=(10,1.5), tight_layout=True)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("-- loading train dataset")
    kwargs = {'num_workers': 20, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        FukudaECGDataset('train', 'partial'),
        #FukudaECGDataset('lead1_small.pickle', True), #, 'partial'),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)

    print("-- loading validation dataset")
    kwargs = {'pin_memory': True}
    valid_arrhythm_loader = torch.utils.data.DataLoader(
        FukudaECGDataset('valid_arrhythm'),
        batch_size=1,
        shuffle=False, **kwargs)
    valid_normal_loader   = torch.utils.data.DataLoader(
        FukudaECGDataset('valid_normal'),
        batch_size=1,
        shuffle=False, **kwargs)

    print("-- building model")
    model        = ConvVAE(args.latent_size)
    model_single = ConvVAE(args.latent_size)
    optimizer    = optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)
    model_single.to(device)

    start_epoch = 0
    best_test_loss = np.finfo('f').max

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint %s' % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s' % args.resume)
        else:
            print('=> no checkpoint found at %s' % args.resume)

    if torch.cuda.device_count() > 1:
        print("-- detected {} GPUs".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    print("-- beginning training ")
    train_loss_history = []
    valid_loss_history = []
    valid_mll_history  = []
    for epoch in range(start_epoch, args.epochs):
        train_loss = train(epoch, model, train_loader, optimizer, args)

        if type(model) == torch.nn.DataParallel:
            state = model.module.state_dict()
        else:
            state = model.state_dict()
        
        model_single.load_state_dict(state)
        test_loss, test_mll = test(epoch, model_single,
                                   valid_normal_loader, valid_arrhythm_loader, args) 


        print('Epoch [%d/%d] loss: %.3f val_loss: %.3f mll: %.3f' %
              (epoch + 1, args.epochs, train_loss, test_loss, test_mll))

        train_loss_history.append(train_loss)
        valid_loss_history.append(test_loss)
        valid_mll_history.append(test_mll)

        is_best = test_loss < best_test_loss
        best_test_loss = min(test_loss, best_test_loss)

        save_checkpoint({
            'epoch':          epoch,
            'test_loss' :     test_loss,
            'test_mll'  :     test_mll,
            'best_test_loss': best_test_loss,
            'state_dict':     state,
            'optimizer':      optimizer.state_dict(),
        }, is_best, os.path.join('./',args.ckpt_dir))

        np.save("train_loss.npy", train_loss_history)
        np.save("valid_loss.npy", valid_loss_history)
        np.save("valid_mll.npy",  valid_mll_history)


if __name__ == '__main__':
    main()
