import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from skimage.transform import resize
import palettable
import seaborn as sns
import torch

plt.rcParams['font.family'] = 'Arial'

########################################################################################
#####################      visualizalize training curve      ###########################
########################################################################################

def set_axes(ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    'Configure Matplotlib axes.'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if legend:
        ax.legend(legend)

class Animator:
    'Plot training curves incrementally.'
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Update multiple curves incrementally.
        if legend is None:
            legend = []
        plt.rcParams['figure.dpi'] = 300  # Set figure resolution.
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Capture axis settings in a lambda.
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add points to each curve.
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        # display.clear_output(wait=True)

########################################################################################
#####################   Evaluate the contact map prediction    #########################
########################################################################################

def resize_image_skimage(image, target_size):
    # Resize the image.
    resized_image = resize(image, target_size, order=0, anti_aliasing=True)
    return resized_image


def evaluate_conmap_dm_dataset(conmap_total_preds_resize, conmap_total_truths, L=1):
    pres = []
    for conmap_pred_resize, conmap_truth in zip(conmap_total_preds_resize, conmap_total_truths):
        
        conmap_pred_resize = conmap_pred_resize.cpu().numpy().squeeze(0)
        
        # print('conmap_pred.shape: ', conmap_pred.shape)
        # print('conmap_truth.shape: ', conmap_truth.shape)
        
        real_shape = conmap_truth.shape
        conmap_pred = resize_image_skimage(conmap_pred_resize, real_shape)

        # print('conmap_pred.shape: ', conmap_pred.shape)
        # print('conmap_truth.shape: ', conmap_truth.shape)
        
        pre = evaluate_conmap_dm(conmap_pred, conmap_truth, L=L)
        pres.append(pre)

    return pres


def evaluate_conmap_dm(conmap_pred, conmap_truth, L=1, dist_cutoff=10):
    'Evaluate precision among the L smallest predicted distances.\n\nFractional L values specify a fraction of the RNA length.'
    # conmap_pred = conmap_pred.cpu().numpy()
    # conmap_truth = conmap_truth.cpu().numpy()
    if L < 1:
        L = round(L * conmap_pred.shape[0])
    if conmap_pred.shape != conmap_truth.shape:
        raise ValueError('Predicted and true contact maps must have the same shape')
    L = min(int(L), conmap_pred.size)
    if L <= 0:
        return 0.0
    sorted_indices = np.argsort(conmap_pred, axis=None)  # Sort distances in ascending order.
    # Select the first L indices.
    top_L_indices = sorted_indices[:L]
    # print(top_L_indices)
 
    # Convert flat indices to row and column coordinates.
    top_L_coords = np.unravel_index(top_L_indices, conmap_pred.shape)
    # print(top_L_coords)

    # Read the corresponding ground-truth values.
    conmap_truth = np.where(conmap_truth <= dist_cutoff, 1, 0)
    top_L_truth_values = conmap_truth[top_L_coords]

    # Compute the fraction of contacts among the top L pairs.
    pre = np.sum(top_L_truth_values) / L  # True Positives / L


    return pre

########################################################################################
#####################    calculate the contact map mse         #########################
########################################################################################

def calculate_mse(matrix1, matrix2):
    'Return the mean squared error between two matrices of equal shape.'
    # Require matching matrix shapes.
    if matrix1.shape != matrix2.shape:
        raise ValueError("两个矩阵的形状必须相同")
    # Convert to PyTorch tensors.
    matrix1 = torch.tensor(matrix1, dtype=torch.float32)
    matrix2 = torch.tensor(matrix2, dtype=torch.float32)
    # print(matrix1, matrix2)
    # Square elementwise differences.
    squared_diff = torch.square(matrix1 - matrix2)
    # Average over all elements.
    mse = torch.mean(squared_diff)
    return mse.item()


def conmap_mses(conmap_total_preds_resize, conmap_total_truths):
    mses = []
    for conmap_pred_resize, conmap_truth in zip(conmap_total_preds_resize, conmap_total_truths):
        
        conmap_pred_resize = conmap_pred_resize.cpu().numpy().squeeze(0)
        
        # print('conmap_pred.shape: ', conmap_pred.shape)
        # print('conmap_truth.shape: ', conmap_truth.shape)
        
        real_shape = conmap_truth.shape
        conmap_pred = resize_image_skimage(conmap_pred_resize, real_shape)

        # print('conmap_pred.shape: ', conmap_pred.shape)
        # print('conmap_truth.shape: ', conmap_truth.shape)
        
        mse = calculate_mse(conmap_pred, conmap_truth)
        mses.append(mse)

    return mses


########################################################################################
#####################        Visualize the RP interface        #########################
########################################################################################
def visual_RP_interface_compare_dataset(conmap_total_preds_resize, conmap_total_truths, pdbnames, path):
    'Save predicted and ground-truth contact-map comparisons for a dataset.'

    for idx, (conmap_pred_resize, conmap_truth) in enumerate(zip(conmap_total_preds_resize, conmap_total_truths)):
        
        conmap_pred_resize = conmap_pred_resize.cpu().numpy().squeeze(0)
        
        # print('conmap_pred.shape: ', conmap_pred.shape)
        # print('conmap_truth.shape: ', conmap_truth.shape)
        
        real_shape = conmap_truth.shape
        conmap_pred = resize_image_skimage(conmap_pred_resize, real_shape)
        pdbname = pdbnames[idx]
        visual_RP_interface_compare(conmap_pred, conmap_truth, pdbname, path)

        # print('conmap_pred.shape: ', conmap_pred.shape)
        # print('conmap_truth.shape: ', conmap_truth.shape)
        


def visual_RP_interface_compare(conmap_pred, conmap_truth, pdbname, path):
    'Save predicted and ground-truth contact maps side by side.'

    cmap = palettable.cmocean.sequential.Haline_20.mpl_colormap
    # cmap = 'coolwarm'

    plt.figure(figsize=(12, 4))  # Set a width suitable for two panels.

    # Plot the predicted contact-map heatmap.
    plt.subplot(1, 2, 1)  # First panel of a one-row, two-column layout.
    # sns.heatmap(1 - conmap_pred, cmap=cmap, annot=False, cbar=False)
    sns.heatmap(conmap_pred, cmap=cmap, annot=False, cbar=False)
    plt.title(f'Predicted Contact Map - {pdbname}')

    # Plot the ground-truth contact-map heatmap.
    plt.subplot(1, 2, 2)  # Second panel of a one-row, two-column layout.
    # sns.heatmap(1 - conmap_truth, cmap=cmap, annot=False, cbar=False)
    sns.heatmap(conmap_truth, cmap=cmap, annot=False, cbar=False)
    plt.title(f'True Contact Map - {pdbname}')

    plt.suptitle(pdbname)  # Set the overall figure title.

    save_file = f'{path}/{pdbname}.png'
    plt.savefig(save_file, dpi=600)
    plt.close()  # Close the figure to release memory.
    
    # Release references to the NumPy arrays.
    del conmap_pred
    del conmap_truth
    
    # Run garbage collection.
    import gc
    gc.collect()

