#import matplotlib
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import argparse
import datetime

## code from https://github.com/VictorBoutin/RiemannEBM/blob/main/utils/monitoring.py

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'T', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'F', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def plot_img(data, nrow=4, ncol=8, padding=2, normalize=True, saving_path=None, title=None, pad_value=0, figsize=(8, 8),
             dpi=100, scale_each=False, cmap=None, axs=None):
    nb_image = nrow * ncol
    data_to_plot = torchvision.utils.make_grid(data[:nb_image], nrow=ncol, padding=padding, normalize=normalize,
                                               pad_value=pad_value, scale_each=scale_each)
    show(data_to_plot.detach().cpu(), saving_path=saving_path, title=title, figsize=figsize, dpi=dpi, cmap=cmap,
         axs=axs)


def make_grid(data, nrow=4, ncol=8, padding=2, normalize=True, pad_value=0, scale_each=False):
    nb_image = nrow * ncol
    data_to_plot = torchvision.utils.make_grid(data[:nb_image], nrow=ncol, padding=padding, normalize=normalize,
                                               pad_value=pad_value, scale_each=scale_each)
    return data_to_plot

def show(img, title=None, saving_path=None, figsize=(8, 8), dpi=100, cmap=None, axs=None):
    npimg = img.numpy()
    if axs is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest', cmap=cmap)
        plt.axis('off')
        if title is not None:
            plt.title(title)
        if saving_path is None:
            plt.show()
        else:
            plt.savefig(saving_path + '/' + title + '.png')
        plt.close()
    else:
        axs.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest', cmap=cmap)

def name_model(args):
    model_name = ""
    if args.device == "meso":
        model_name += "CCV"
    data_time = str(datetime.datetime.now())[0:19].replace(' ', '_')
    data_time = data_time.replace('-','')
    model_name += '_' + data_time.replace(':', '')
    if hasattr(args, 'db_type'):
        if args.db_type == "stable_diff_14_aug":
            model_name += f'_AUG'
        elif args.db_type == "stable_diff_14_aug_max":
            model_name += f'_MaxAUG'
        elif args.db_type == "stable_diff_14_aug_max_max":
            model_name += f'_MMaxAUG'
        elif args.db_type == "stable_diff_14_aug_max_rot":
            model_name += f'_RotAUG'
    if hasattr(args, 'training'):
        model_name += f'_{args.training}'
    if hasattr(args, 'energy_func'):
        model_name += f'_{args.energy_func}'
    if hasattr(args, 'multiplier'):
        model_name += f'_x{args.multiplier}'
    if hasattr(args, 'dsm_weight'):
        model_name += f'_w_dsm{args.dsm_weight}'
    if hasattr(args, f'n_steps'):
        model_name += f'_Stp{args.n_steps}'
    if hasattr(args, 'sgld_lr'):
        model_name += f'_SgldLr={args.sgld_lr}'
    if hasattr(args, 'lr_init'):
        model_name += f'_LR{args.lr_init:0.1e}'
    if hasattr(args, 'w_regul'):
        model_name += f'_Wr{args.w_regul:0.1e}'
    if hasattr(args, 'gamma_scheduler'):
        if args.gamma_scheduler != 0:
            model_name += f'_GSched{args.gamma_scheduler:0.1e}'
    if hasattr(args, 'spec_norm'):
        if args.spec_norm != 0:
            model_name += f'_SN_Clip{args.gradient_clip:0.1e}'
    if hasattr(args, 'grad_regul'):
        if args.grad_regul != 0:
            model_name += f'_GradReg{args.grad_regul:0.1e}Noise{args.noise_grad_regul:0.1e}'
    if hasattr(args, 'w_last_gradient'):
        if args.w_last_gradient != 0:
            model_name += f'_LastReg{args.w_last_gradient:0.1e}'
    return model_name


def name_model_interp(args):
    model_name = ""
    if args.device == "meso":
        model_name += "CCV"
    data_time = str(datetime.datetime.now())[0:19].replace(' ', '_')
    data_time = data_time.replace('-','')
    model_name += '_' + data_time.replace(':', '')

    model_name += f'_{args.metric}'
    model_name += f'_COEFF{args.coeff}'
    model_name += f'_{args.interp_type}'
    model_name += f'_ALPHA{args.alpha}'
    model_name += f'_ACCUM{args.num_accum}'
    model_name += f'_STEP{args.nb_steps}'
    model_name += f'_CH{args.num_channels}'
    model_name += f'_BS{args.batch_size}'

    return model_name


def name_model_interp2(args):
    model_name = ""
    if args.device == "meso":
        model_name += "CCV"
    data_time = str(datetime.datetime.now())[0:19].replace(' ', '_')
    data_time = data_time.replace('-','')
    model_name += '_' + data_time.replace(':', '')

    model_name += f'_{args.metric}'
    if args.rot_dist == "gaussian":
        model_name += f'_GAUSS'
    if args.ebm_multiplier is not None:
        model_name += f'_MultEBM{args.ebm_multiplier}'
    if args.gamma_land is not None:
        model_name += f'_GammaLand{args.gamma_land}'
    if args.rbf_center is not None and args.rbf_kappa:
        model_name += f'_KaRBF{args.rbf_kappa}_CenRBF{args.rbf_center}'

    model_name += f'_MinH{args.min_h}'
    model_name += f'_MaxH{args.max_h}'
    model_name += f'_STEP{args.t_steps}'
    model_name += f'_BS{args.batch_size}'
    model_name += f'_CHAN{args.num_channels}'

    return model_name