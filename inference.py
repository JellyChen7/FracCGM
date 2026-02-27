import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import imageio
import numpy as np
import torch, h5py
from modules import UNet_conditional
from ddim import Diffusion
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.stats import norm

def set_nature_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (3.54, 2.65),  # Nature single column width
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False
    })

set_nature_style()


def visualize_temperature_comparison(temp_pred, temp_true, temp_pred_std, timesteps=None, 
                                    save_pdf=True, save_dir='./results', 
                                    filename='temperature_comparison', dpi=600):
    """
    Visualize and compare temperature prediction result
    
    Parameters:
    temp_pred: numpy array of shape [11, 128, 128] - Predict temperature
    temp_true: numpy array of shape [11, 128, 128] - True temperature
    timesteps: Time step information
    save_pdf: bool, save PDF file 
    save_dir: str, Directory
    filename: str, File name
    dpi: int, Image resolution
    """
    
    if timesteps is None:
        timesteps = range(11)
    
    n_timesteps = temp_pred.shape[0]
    
    fig = plt.figure(figsize=(7.2, 6))  # Nature double column width
    gs = GridSpec(5, n_timesteps, figure=fig, height_ratios=[1, 1, 1, 1, 0.2],
                 hspace=0.1, wspace=0.4)
    
    error = temp_pred - temp_true
    abs_error = np.abs(error)
    
    global_stats = {
        'mse': np.mean(error**2),
        'rmse': np.sqrt(np.mean(error**2)),
        'mae': np.mean(abs_error),
        'max_abs_error': np.max(abs_error),
        'correlation': pearsonr(temp_pred.flatten(), temp_true.flatten())[0]
    }
    
    vmin_true = np.min(temp_true)
    vmax_true = np.max(temp_true)
    vmin_pred = np.min(temp_pred)
    vmax_pred = np.max(temp_pred)
    vmin = min(vmin_true, vmin_pred)
    vmax = max(vmax_true, vmax_pred)
    
    vmax_abs_error = 1  
    
    for i in range(n_timesteps):
        ax1 = fig.add_subplot(gs[0, i])
        im1 = ax1.imshow(temp_true[i], cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='equal')
        ax1.set_title(f't={timesteps[i]*360} days', fontsize=9, pad=4)
        ax1.set_xticks([])
        ax1.set_yticks([])
        if i == 0:
            ax1.set_ylabel('True', fontsize=9, labelpad=2)
        if i == 5:
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02, shrink=0.8)
        
        ax2 = fig.add_subplot(gs[1, i])
        im2 = ax2.imshow(temp_pred[i], cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='equal')
        # ax2.set_title(f't={timesteps[i]}', fontsize=9, pad=4)
        ax2.set_xticks([])
        ax2.set_yticks([])
        if i == 0:
            ax2.set_ylabel('Predicted', fontsize=9, labelpad=2)
        if i == 5:
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02, shrink=0.8)
        
        ax3 = fig.add_subplot(gs[2, i])
        im3 = ax3.imshow(abs_error[i], cmap='PuBu', vmin=0, vmax=vmax_abs_error, aspect='equal')
        # ax3.set_title(f't={timesteps[i]}', fontsize=9, pad=4)
        ax3.set_xticks([])
        ax3.set_yticks([])
        if i == 0:
            ax3.set_ylabel('Error', fontsize=9, labelpad=2)
        if i == 5:
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02, shrink=0.8)

        ax4 = fig.add_subplot(gs[3, i])
        im4 = ax4.imshow(temp_pred_std[i], cmap='PuBu', vmin=0, vmax=vmax_abs_error, aspect='equal')
        # ax4.set_title(f't={timesteps[i]}', fontsize=9, pad=4)
        ax4.set_xticks([])
        ax4.set_yticks([])
        if i == 0:
            ax4.set_ylabel('Standard deviation', fontsize=9, labelpad=2)
        if i == 5:
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.02, shrink=0.8)
    ax_stats = fig.add_subplot(gs[4, :])
    ax_stats.axis('off')
    
    stats_text = (
        f"MSE: {global_stats['mse']:.3f} | "
        f"RMSE: {global_stats['rmse']:.3f} | "
        f"MAE: {global_stats['mae']:.3f} | "
        f"R: {global_stats['correlation']:.3f}"
    )
    
    ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center', 
                 fontsize=9, transform=ax_stats.transAxes)
    
    plt.tight_layout()
    
    if save_pdf:
        save_figure_as_pdf(fig, filename, save_dir, dpi=dpi)
    
    plt.show()
    
    return global_stats

def plot_timeseries_metrics(temp_pred, temp_true, save_pdf=True, 
                           save_dir='./results', filename='timeseries_metrics', dpi=600):
    """
    
    Parameters:
    temp_pred: numpy array of shape [11, 128, 128] 
    temp_true: numpy array of shape [11, 128, 128]
    save_pdf: bool
    save_dir: str
    filename: str
    dpi: int
    """
    n_timesteps = temp_pred.shape[0]
    
    mse_per_step = []
    mae_per_step = []
    correlation_per_step = []
    
    for i in range(n_timesteps):
        pred_flat = temp_pred[i].flatten()
        true_flat = temp_true[i].flatten()
        
        mse_per_step.append(np.mean((pred_flat - true_flat) ** 2))
        mae_per_step.append(np.mean(np.abs(pred_flat - true_flat)))
        correlation_per_step.append(pearsonr(pred_flat, true_flat)[0])
    
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    axes[0, 0].plot(range(n_timesteps), mse_per_step, 'o-', linewidth=1.5, 
                   markersize=3, color=colors[0], markeredgecolor='white', markeredgewidth=0.5)
    axes[0, 0].set_xlabel('Time step')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_xticks(range(0, n_timesteps, 2))
    
    axes[0, 1].plot(range(n_timesteps), mae_per_step, 's-', linewidth=1.5, 
                   markersize=3, color=colors[1], markeredgecolor='white', markeredgewidth=0.5)
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_xticks(range(0, n_timesteps, 2))
    
    axes[1, 0].plot(range(n_timesteps), correlation_per_step, '^-', linewidth=1.5, 
                   markersize=3, color=colors[2], markeredgecolor='white', markeredgewidth=0.5)
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].set_ylabel('Pearson R')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xticks(range(0, n_timesteps, 2))
    
    y_true = temp_true.flatten()
    y_pred = temp_pred.flatten()

    xy = np.vstack([y_true, y_pred])
    z = gaussian_kde(xy)(xy)
    
    p_low = np.percentile(z, 5)
    p_high = np.percentile(z, 95)
    z_normalized = np.clip((z - p_low) / (p_high - p_low) * 100, 0, 100)

    sc = axes[1, 1].scatter(y_true, y_pred, c=z_normalized, cmap='plasma', 
                           alpha=0.8, s=1, vmin=0, vmax=1)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'k--', 
                   linewidth=1, alpha=0.7, label='y=x')
    
    cbar = plt.colorbar(sc, ax=axes[1, 1], shrink=0.8)
    cbar.set_label('Density (%)', rotation=270, labelpad=10)
    cbar.set_ticks([0, 50, 100])
    
    axes[1, 1].set_xlabel('True values')
    axes[1, 1].set_ylabel('Predicted values')
    axes[1, 1].legend(frameon=False, fontsize=8)
    
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_pdf:
        save_figure_as_pdf(fig, filename, save_dir, dpi=dpi)
    
    plt.show()
    
    return {
        'mse_per_step': mse_per_step,
        'mae_per_step': mae_per_step,
        'correlation_per_step': correlation_per_step
    }

def plot_spatial_error_distribution(temp_pred, temp_true, save_pdf=True, 
                                   save_dir='./results', filename='spatial_error_distribution', dpi=600):
    """
    
    Parameters:
    temp_pred: numpy array of shape [11, 128, 128] 
    temp_true: numpy array of shape [11, 128, 128]
    save_pdf: bool
    save_dir: str
    filename: str
    dpi: int
    """
    error = temp_pred - temp_true
    abs_error = np.abs(error)
    
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))
    axes = axes.flatten()
    
    axes[0].hist(error.flatten(), bins=30, alpha=0.8, color='#1f77b4', 
                edgecolor='white', linewidth=0.5)
    axes[0].set_xlabel('Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim(-0.5, 0.5)
    axes[0].axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    axes[1].hist(abs_error.flatten(), bins=30, alpha=0.8, color='#d62728', 
                edgecolor='white', linewidth=0.5)
    axes[1].set_xlabel('Absolute error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim(0, 0.5)
    
    mean_abs_error = np.mean(abs_error, axis=0)
    im0 = axes[2].imshow(mean_abs_error, cmap='Reds', aspect='equal')
    axes[2].set_title('Mean error', fontsize=9)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.colorbar(im0, ax=axes[2], fraction=0.046, pad=0.02, shrink=0.8)
    
    spatial_corr = np.zeros((128, 128))
    for i in range(128):
        for j in range(128):
            spatial_corr[i, j] = pearsonr(temp_pred[:, i, j], temp_true[:, i, j])[0]
    
    im1 = axes[3].imshow(spatial_corr, cmap='coolwarm', vmin=0, vmax=1, aspect='equal')
    axes[3].set_title('Spatial R', fontsize=9)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    plt.colorbar(im1, ax=axes[3], fraction=0.046, pad=0.02, shrink=0.8)
    
    max_error_timestep = np.argmax(np.max(abs_error, axis=(1, 2)))
    im2 = axes[4].imshow(abs_error[max_error_timestep], cmap='Reds', aspect='equal')
    axes[4].set_title(f'Max error (t={max_error_timestep})', fontsize=9)
    axes[4].set_xticks([])
    axes[4].set_yticks([])
    plt.colorbar(im2, ax=axes[4], fraction=0.046, pad=0.02, shrink=0.8)
    
    min_error_timestep = np.argmin(np.mean(abs_error, axis=(1, 2)))
    im3 = axes[5].imshow(abs_error[min_error_timestep], cmap='Reds', aspect='equal')
    axes[5].set_title(f'Min error (t={min_error_timestep})', fontsize=9)
    axes[5].set_xticks([])
    axes[5].set_yticks([])
    plt.colorbar(im3, ax=axes[5], fraction=0.046, pad=0.02, shrink=0.8)
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_pdf:
        save_figure_as_pdf(fig, filename, save_dir, dpi=dpi)
    
    plt.show()

def save_figure_as_pdf(fig, filename, save_dir='./results', dpi=600, metadata=None):
    """
    
    Parameters:
    fig: matplotlib Figure
    filename: str
    save_dir: str
    dpi: int
    metadata: dict
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if metadata is None:
        metadata = {
            'Creator': 'Temperature Visualization Script',
            'CreationDate': datetime.now(),
            'Title': filename
        }
    
    filename = os.path.splitext(filename)[0]
    pdf_path = os.path.join(save_dir, f"{filename}.pdf")
    
    fig.savefig(
        pdf_path,
        format='pdf',
        dpi=dpi,
        bbox_inches='tight',
        metadata=metadata
    )
    
    print(f"✓ Figure saved as PDF: {pdf_path}")
    return pdf_path

device = "cpu"
model = UNet_conditional(1, 3, device=device)
diffusion = Diffusion(img_size=128, device=device)

ckpt = torch.load("ckpt2000.pt", map_location=torch.device('cpu'))
model.load_state_dict(ckpt)

test_index = 800
n_posterior = 20
num_test = 100

frac_data = h5py.File('Binary_training.mat','r')
fracture = frac_data['Binary']
fracture = np.array(fracture).transpose((2,1,0))
fracture = fracture[test_index:test_index+num_test,np.newaxis,:,:]
fracture = np.repeat(fracture, 11, axis=1)
frac_data.close()

pred_data = h5py.File('state_training.mat','r')
state = pred_data['state_training']
state = np.array(state).transpose((3,0,2,1))
state = state[test_index:test_index+num_test,:,:,:]
pred_data.close()

t = np.linspace(-1, 1, 11)
t = t[np.newaxis, :, np.newaxis, np.newaxis]
t_step = np.repeat(t, num_test, axis=0)
t_step = np.repeat(t_step, 128, axis=2)
t_step = np.repeat(t_step, 128, axis=3)
    
fracture = fracture.reshape(-1, 11, 1, 128, 128)
t_step = t_step.reshape(-1, 11, 1, 128, 128)
state = state.reshape(-1, 11, 1, 128, 128)
# 2
index = 0
fracture_index = fracture[index:index+1]
t_index = t_step[index:index+1]
fracture_index = np.repeat(fracture_index, n_posterior, axis=0)
t_index = np.repeat(t_index, n_posterior, axis=0)
temp_pred = np.zeros((11, 128, 128))

x_denoising = torch.empty((n_posterior, 11, 20, 128, 128)).to(device)
for i in range(11):
    frac = fracture_index[:, i]
    t = t_index[:, i]
    y = np.concatenate((frac, t), 1)
    y = torch.FloatTensor(y).to(device)
    x, x_array = diffusion.sample_ddim(model, len(y), y, ddim_timesteps=20)
    x_denoising[:,i,:,:,:] = x_array[:,0,:,:,:]
    x = x.clamp(-1, 1).cpu().numpy()

    temp_pred_mean = np.mean(x, 0)[0]

    temp_pred[i] = temp_pred_mean

fracture_select = fracture_index[0,0,0]
temp_true = state[index, :, 0]

temp_pred_std = torch.std(x_denoising[:,:,-1,:,:],dim=0).numpy()

plt.figure(figsize=(3.54, 2.65))
plt.imshow(fracture_select, cmap='Greys', aspect='equal')
plt.axis('off')
plt.tight_layout()
plt.show()

# ======================
plt.rcParams.update({
    'font.family': 'Arial',  
    'font.size': 8,
    'font.weight': 'normal',
    
    'axes.linewidth': 0.5,  
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
    
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.25,
    'ytick.minor.width': 0.25,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    
    'legend.fontsize': 7,
    'legend.frameon': False,
    
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    
    'figure.dpi': 600,  
    'savefig.dpi': 600,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    
    'axes.prop_cycle': plt.cycler(color=plt.cm.Set2(np.linspace(0, 1, 8))),
})

timesteps = list(reversed(range(1, 1000, 50)))
cycle_steps = list(range(0, 3601, 360))

subplot_config = [
    {'data_idx': 0, 'cycle_idx': 0, 'grid_pos': (0, 0)},
    {'data_idx': 2, 'cycle_idx': 2, 'grid_pos': (0, 1), 'show_title': True},
    {'data_idx': 4, 'cycle_idx': 4, 'grid_pos': (0, 2)},
    {'data_idx': 6, 'cycle_idx': 6, 'grid_pos': (1, 0)},
    {'data_idx': 8, 'cycle_idx': 8, 'grid_pos': (1, 1)},
    {'data_idx': 10, 'cycle_idx': 10, 'grid_pos': (1, 2)},
]

for i in range(20):
    fig = plt.figure(figsize=(7.2, 4.5))  
    gs = GridSpec(3, 4, figure=fig, 
                  height_ratios=[1, 1, 0.08],
                  width_ratios=[1, 1, 1, 0.05],
                  hspace=0.15, wspace=0.12)
    
    for idx, config in enumerate(subplot_config):
        row, col = config['grid_pos']
        ax = fig.add_subplot(gs[row, col])
        
        data = x_denoising[0, config['data_idx'], i].clamp(0, 1).cpu().numpy()
        
        im = ax.imshow(data, 
                      cmap='RdYlBu_r',  # 或 'viridis', 'plasma'
                      vmin=0, 
                      vmax=1, 
                      aspect='equal',
                      interpolation='bilinear')  # 添加平滑插值
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.set_xlabel(f't = {cycle_steps[config["cycle_idx"]]}',
                     fontsize=7,
                     labelpad=2)
        
        if config.get('show_title', False):
            ax.set_title(f'T = {timesteps[i]}',
                        fontsize=8,
                        pad=6,
                        fontweight='medium')
        
        label_text = f'({chr(97+idx)})' 
        ax.text(0.02, 0.98, label_text,
               transform=ax.transAxes,
               fontsize=8,
               fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.2', 
                        facecolor='white', 
                        alpha=0.8,
                        linewidth=0.5))
    
    cbar_ax = fig.add_subplot(gs[2, :3])  
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Temperature (normalized)', 
                  fontsize=7,
                  labelpad=2)
    cbar.ax.tick_params(labelsize=6, width=0.5)
    
    text_ax = fig.add_subplot(gs[:2, 3])
    text_ax.axis('off')
    
    info_text = f'Frame: {i+1}/20\nTimestep: {timesteps[i]}\n\nColor scale:\n0.0 (min) to 1.0 (max)'
    text_ax.text(0.1, 0.5, info_text,
                transform=text_ax.transAxes,
                fontsize=6,
                verticalalignment='center',
                linespacing=1.5)
    
    fig.suptitle('Temporal Evolution of Temperature Field',
                fontsize=10,
                fontweight='medium',
                y=0.98)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])  
    
    plt.savefig(f'./result/sampled_temp_{i+1:02d}.png',
                dpi=600,
                bbox_inches='tight',
                pad_inches=0.05,
                facecolor='white',
                edgecolor='none')
    
    plt.close(fig)
    print(f'Saved frame {i+1}/20')

print("\nCreating GIF animation...")
images = []

for i in range(1, 21):
    filename = f'./result/sampled_temp_{i:02d}.png'
    images.append(imageio.imread(filename))

imageio.mimsave(
    './result/temperature_evolution_nature_style.gif',
    images,
    duration=0.3, 
    fps=5,      
    loop=0,       
    subrectangles=True
)

print("GIF animation saved successfully!")


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm

plt.rcParams.update({
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'legend.title_fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.autolayout': True,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.3,
    'lines.linewidth': 1.2,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
})

colors = {
    'true': '#2E4057',       
    'pred': '#D1495B',        
    'uncertainty': '#D1495B', 
    'grid': '#E6E6E6',        
    'background': '#FFFFFF',  
    'text': '#2C3E50',       
    'highlight1': '#28AFB0',  
    'highlight2': '#F4A261', 
    'highlight3': '#6A994E', 
}


points = [(13, 15), (13, 115), (62, 62), (115, 13), (113, 113)]
point_labels = ['(13, 13)', '(13, 115)', '(64, 64)', '(115, 13)', '(115, 115)']



from scipy.interpolate import make_interp_spline
import numpy as np

fig, axes = plt.subplots(2, 3, figsize=(7.5, 5))
axes = axes.flat

for idx, ((row, col), label) in enumerate(zip(points, point_labels)):
    ax = axes[idx]
    
    ax.set_facecolor(colors['background'])
    
    true_vals = temp_true[:, row, col]
    pred_vals = temp_pred[:, row, col]
    pred_stds = temp_pred_std[:, row, col]
    
    time_points = np.arange(len(true_vals))
    
    n_points = 200  
    time_points_dense = np.linspace(0, len(true_vals)-1, n_points)
    
    if len(time_points) >= 4:  
        spline_true = make_interp_spline(time_points, true_vals, k=3)
        true_vals_dense = spline_true(time_points_dense)
    else:
        true_vals_dense = np.interp(time_points_dense, time_points, true_vals)
    
    if len(time_points) >= 4:
        spline_pred = make_interp_spline(time_points, pred_vals, k=3)
        pred_vals_dense = spline_pred(time_points_dense)
    else:
        pred_vals_dense = np.interp(time_points_dense, time_points, pred_vals)
    
    upper_bound = pred_vals + 2*pred_stds
    lower_bound = pred_vals - 2*pred_stds
    
    if len(time_points) >= 4:
        spline_upper = make_interp_spline(time_points, upper_bound, k=3)
        upper_dense = spline_upper(time_points_dense)
        spline_lower = make_interp_spline(time_points, lower_bound, k=3)
        lower_dense = spline_lower(time_points_dense)
    else:
        upper_dense = np.interp(time_points_dense, time_points, upper_bound)
        lower_dense = np.interp(time_points_dense, time_points, lower_bound)
    
    upper_1sigma = pred_vals + pred_stds
    lower_1sigma = pred_vals - pred_stds
    
    if len(time_points) >= 4:
        spline_upper1 = make_interp_spline(time_points, upper_1sigma, k=3)
        upper1_dense = spline_upper1(time_points_dense)
        spline_lower1 = make_interp_spline(time_points, lower_1sigma, k=3)
        lower1_dense = spline_lower1(time_points_dense)
    else:
        upper1_dense = np.interp(time_points_dense, time_points, upper_1sigma)
        lower1_dense = np.interp(time_points_dense, time_points, lower_1sigma)
    
    ax.fill_between(time_points_dense, 
                    lower_dense, 
                    upper_dense, 
                    color=colors['pred'], alpha=0.15, 
                    label='±2σ uncertainty', zorder=2)
    
    ax.fill_between(time_points_dense, 
                    lower1_dense, 
                    upper1_dense, 
                    color=colors['pred'], alpha=0.25, 
                    label='±1σ uncertainty', zorder=3)
    
    ax.plot(time_points_dense, true_vals_dense, color=colors['true'], 
            linewidth=1.8, label='True temperature', 
            marker='o', markersize=2.5, markevery=20, zorder=4)
    
    ax.plot(time_points_dense, pred_vals_dense, color=colors['pred'], 
            linewidth=1.8, label='Predicted mean', 
            marker='s', markersize=2.5, markevery=20, zorder=4)
    
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(true_vals)-1)
    
    ax.grid(True, color=colors['grid'], alpha=0.4, 
            linestyle='-', linewidth=0.5, zorder=1)
    
    ax.set_title(f'Location {label}', fontsize=7.5, pad=4, 
                 color=colors['text'], fontweight='medium')
    ax.set_xlabel('Time step', fontsize=8, labelpad=3, color=colors['text'])
    ax.set_ylabel('Temperature', fontsize=8, labelpad=3, color=colors['text'])
    
    ax.tick_params(axis='both', colors=colors['text'])

ax = axes[5]
ax.set_facecolor(colors['background'])
ax.set_axis_off()

legend_elements = [
    mpl.lines.Line2D([0], [0], color=colors['true'], lw=1.8, 
                    marker='o', markersize=4, label='True temperature'),
    mpl.lines.Line2D([0], [0], color=colors['pred'], lw=1.8, 
                    marker='s', markersize=4, label='Predicted mean'),
    mpl.patches.Patch(facecolor=mpl.colors.to_rgba(colors['pred'], 0.25), 
                     edgecolor=colors['pred'], alpha=0.5, 
                     label='±1σ uncertainty'),
    mpl.patches.Patch(facecolor=mpl.colors.to_rgba(colors['pred'], 0.15), 
                     edgecolor=colors['pred'], alpha=0.3, 
                     label='±2σ uncertainty'),
]

legend = ax.legend(handles=legend_elements, loc='center left', 
                  frameon=True, framealpha=0.95, 
                  edgecolor=colors['grid'])
legend.get_frame().set_linewidth(0.5)

plt.suptitle('Temperature Distribution Analysis at Production Wells', 
             fontsize=9, fontweight='bold', y=0.98, color=colors['text'])

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('temperature_distribution_analysis.pdf', format='pdf', 
            bbox_inches='tight', facecolor=colors['background'])

plt.show()



stats = visualize_temperature_comparison(temp_pred[::2,:,:], temp_true[::2,:,:],temp_pred_std[::2,:,:])

time_metrics = plot_timeseries_metrics(temp_pred[::2,:,:], temp_true[::2,:,:])
    
plot_spatial_error_distribution(temp_pred[::2,:,:], temp_true[::2,:,:])

print("\n" + "="*50)
print("Statistical Summary:")
print("="*50)
print(f"Global MSE: {stats['mse']:.4f}")
print(f"Global RMSE: {stats['rmse']:.4f}")
print(f"Global MAE: {stats['mae']:.4f}")
print(f"Max absolute error: {stats['max_abs_error']:.4f}")
print(f"Pearson R: {stats['correlation']:.4f}")

print(f"\nTime-step averages:")
print(f"MSE: {np.mean(time_metrics['mse_per_step']):.4f}")
print(f"MAE: {np.mean(time_metrics['mae_per_step']):.4f}")
print(f"Pearson R: {np.mean(time_metrics['correlation_per_step']):.4f}")



temp_fracture = x_denoising[0,4]
temp_fracture = temp_fracture.numpy()

plt.rcParams.update({
    'font.family': 'Arial',  
    'font.size': 8,          
    'axes.titlesize': 10,    
    'axes.labelsize': 9,    
    'xtick.labelsize': 8,    
    'ytick.labelsize': 8,    
    'legend.fontsize': 8,   
    'figure.titlesize': 12,  
    'lines.linewidth': 1.5, 
    'lines.markersize': 4, 
    'axes.linewidth': 0.8,  
    'grid.linewidth': 0.4,  
    'grid.alpha': 0.3,      
    'savefig.dpi': 600,    
    'savefig.bbox': 'tight', 
    'savefig.pad_inches': 0.1, 
})

provided_colors = np.array([
    [0.267004, 0.004874, 0.329415, 1.      ],
    [0.280894, 0.078907, 0.402329, 1.      ],
    [0.28229 , 0.145912, 0.46151 , 1.      ],
    [0.270595, 0.214069, 0.507052, 1.      ],
    [0.250425, 0.27429 , 0.533103, 1.      ],
    [0.223925, 0.334994, 0.548053, 1.      ],
    [0.19943 , 0.387607, 0.554642, 1.      ],
    [0.175841, 0.44129 , 0.557685, 1.      ],
    [0.15627 , 0.489624, 0.557936, 1.      ],
    [0.136408, 0.541173, 0.554483, 1.      ],
    [0.121831, 0.589055, 0.545623, 1.      ],
    [0.12478 , 0.640461, 0.527068, 1.      ],
    [0.162016, 0.687316, 0.499129, 1.      ],
    [0.239374, 0.735588, 0.455688, 1.      ],
    [0.335885, 0.777018, 0.402049, 1.      ],
    [0.458674, 0.816363, 0.329727, 1.      ],
    [0.585678, 0.846661, 0.249897, 1.      ],
    [0.730889, 0.871916, 0.156029, 1.      ],
    [0.866013, 0.889868, 0.095953, 1.      ],
    [0.993248, 0.906157, 0.143936, 1.      ]
])

nature_colors = provided_colors[:, :3]  

print(f"Color array shape: {nature_colors.shape}")  

def calculate_skewness(data):
    n = len(data)
    if n > 2:
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val > 0:
            return np.mean(((data - mean_val) / std_val) ** 3)
    return 0

def calculate_kurtosis(data):
    n = len(data)
    if n > 3:
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val > 0:
            return np.mean(((data - mean_val) / std_val) ** 4) - 3
    return -3


print("Saving Figure 1: Comprehensive comparison...")
fig = plt.figure(figsize=(7.2, 7.2))

ax1 = plt.subplot(2, 2, 1)
for i in range(20):
    temp_flat = temp_fracture[i].flatten()
    sns.kdeplot(temp_flat, ax=ax1, color=nature_colors[i], alpha=0.7, linewidth=1.0)

ax1.set_xlabel('Temperature (K)', fontsize=9)
ax1.set_ylabel('Probability density', fontsize=9)
ax1.set_title('KDE comparison', fontsize=10, fontweight='bold', pad=10)

legend_elements = [
    Line2D([0], [0], color=nature_colors[0], lw=1.5, label='Sample 1'),
    Line2D([0], [0], color=nature_colors[10], lw=1.5, label='Sample 10'),
    Line2D([0], [0], color=nature_colors[19], lw=1.5, label='Sample 20')
]
ax1.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.9, fancybox=False)

ax2 = plt.subplot(2, 2, 2)
box_data = [temp_fracture[i].flatten() for i in range(20)]
box = ax2.boxplot(box_data, patch_artist=True, showfliers=True, 
                 flierprops=dict(marker='o', markersize=3, alpha=0.5))

for i, patch in enumerate(box['boxes']):
    patch.set_facecolor(nature_colors[i])
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(0.5)

ax2.set_xlabel('Sample index', fontsize=9)
ax2.set_ylabel('Temperature (K)', fontsize=9)
ax2.set_title('Box plot comparison', fontsize=10, fontweight='bold', pad=10)
ax2.set_xticklabels([f'{i+1}' for i in range(20)], rotation=45, fontsize=7)

ax3 = plt.subplot(2, 2, 3)
means = [temp_fracture[i].flatten().mean() for i in range(20)]
stds = [temp_fracture[i].flatten().std() for i in range(20)]

x = np.arange(20)
for i in range(20):
    ax3.errorbar(x[i], means[i], yerr=stds[i], fmt='o', 
                 ecolor=nature_colors[i],  
                 elinewidth=1.0, capsize=3, capthick=1.0, 
                 markersize=4, alpha=0.8,
                 markerfacecolor=nature_colors[i],
                 markeredgecolor='black')

ax3.set_xlabel('Sample index', fontsize=9)
ax3.set_ylabel('Temperature (K)', fontsize=9)
ax3.set_title('Mean ± s.d.', fontsize=10, fontweight='bold', pad=10)
ax3.set_xticks(x)
ax3.set_xticklabels([f'{i+1}' for i in range(20)], fontsize=7, rotation=45)

ax4 = plt.subplot(2, 2, 4)
violin_parts = ax4.violinplot(box_data, showmeans=True, showextrema=True)

for i, pc in enumerate(violin_parts['bodies']):
    pc.set_facecolor(nature_colors[i])
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')
    pc.set_linewidth(0.5)

violin_parts['cmeans'].set_color('white')
violin_parts['cmeans'].set_linewidth(1.5)
violin_parts['cmins'].set_color('black')
violin_parts['cmaxes'].set_color('black')
violin_parts['cbars'].set_color('black')

ax4.set_xlabel('Sample index', fontsize=9)
ax4.set_ylabel('Temperature (K)', fontsize=9)
ax4.set_title('Violin plot comparison', fontsize=10, fontweight='bold', pad=10)
ax4.set_xticks(range(1, 21))
ax4.set_xticklabels([f'{i}' for i in range(1, 21)], fontsize=7, rotation=45)

plt.suptitle('Comparison of temperature distributions across 20 samples', 
             fontsize=11, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Figure1_comprehensive_comparison.pdf', dpi=600, bbox_inches='tight')
plt.close()

print("Saving Figure 2: Detailed distribution analysis...")
fig, axes = plt.subplots(4, 5, figsize=(9.0, 9.0))
axes = axes.flatten()

for i in range(20):
    temp_flat = temp_fracture[i].flatten()
    ax = axes[i]
    
    n, bins, patches = ax.hist(temp_flat, bins=25, density=True, alpha=0.7, 
                               color=nature_colors[i], edgecolor='black', linewidth=0.5)
    
    kde = gaussian_kde(temp_flat)
    x_range = np.linspace(temp_flat.min(), temp_flat.max(), 200)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=1.0, alpha=0.8)
    
    mean_val = temp_flat.mean()
    median_val = np.median(temp_flat)
    
    ax.axvline(mean_val, color='blue', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.axvline(median_val, color='green', linestyle=':', linewidth=1.0, alpha=0.8)
    
    ax.set_title(f'Sample {i+1}', fontsize=9, fontweight='bold', pad=5)
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    
    if i % 5 != 0:
        ax.set_ylabel('')
    if i % 5 != 4:
        ax.set_yticklabels([])
    
    if i < 15:
        ax.set_xlabel('')
        ax.set_xticklabels([])
    
    stats_text = f'Mean: {mean_val:.1f}\nMedian: {median_val:.1f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, 
                     edgecolor='gray', linewidth=0.5))

plt.suptitle('Detailed distribution analysis of 20 temperature samples', 
             fontsize=11, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Figure2_detailed_distribution_analysis.pdf', dpi=600, bbox_inches='tight')
plt.close()

print("Saving Figure 3: Statistical metrics heatmaps...")
stats_data = np.zeros((20, 4))
for i in range(20):
    temp_flat = temp_fracture[i].flatten()
    stats_data[i, 0] = temp_flat.mean()
    stats_data[i, 1] = temp_flat.std()
    stats_data[i, 2] = calculate_skewness(temp_flat)
    stats_data[i, 3] = calculate_kurtosis(temp_flat)

fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2))

im0 = axes[0, 0].imshow(stats_data[:, 0].reshape(4, 5), cmap='viridis', aspect='auto')
axes[0, 0].set_title('Mean values', fontsize=10, fontweight='bold')
cbar0 = plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
cbar0.set_label('Temperature (K)', fontsize=8)
cbar0.ax.tick_params(labelsize=7)

for i in range(4):
    for j in range(5):
        idx = i * 5 + j
        axes[0, 0].text(j, i, f'{stats_data[idx, 0]:.1f}', 
                       ha='center', va='center', 
                       color='white' if stats_data[idx, 0] > np.median(stats_data[:, 0]) else 'black',
                       fontsize=7, fontweight='bold')

im1 = axes[0, 1].imshow(stats_data[:, 1].reshape(4, 5), cmap='plasma', aspect='auto')
axes[0, 1].set_title('Standard deviation', fontsize=10, fontweight='bold')
cbar1 = plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
cbar1.set_label('Temperature (K)', fontsize=8)
cbar1.ax.tick_params(labelsize=7)

for i in range(4):
    for j in range(5):
        idx = i * 5 + j
        axes[0, 1].text(j, i, f'{stats_data[idx, 1]:.1f}', 
                       ha='center', va='center', 
                       color='white' if stats_data[idx, 1] > np.median(stats_data[:, 1]) else 'black',
                       fontsize=7, fontweight='bold')

im2 = axes[1, 0].imshow(stats_data[:, 2].reshape(4, 5), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
axes[1, 0].set_title('Skewness', fontsize=10, fontweight='bold')
cbar2 = plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
cbar2.set_label('Skewness', fontsize=8)
cbar2.ax.tick_params(labelsize=7)

for i in range(4):
    for j in range(5):
        idx = i * 5 + j
        axes[1, 0].text(j, i, f'{stats_data[idx, 2]:.2f}', 
                       ha='center', va='center', 
                       color='white' if abs(stats_data[idx, 2]) > 0.5 else 'black',
                       fontsize=7, fontweight='bold')

im3 = axes[1, 1].imshow(stats_data[:, 3].reshape(4, 5), cmap='coolwarm', aspect='auto')
axes[1, 1].set_title('Kurtosis', fontsize=10, fontweight='bold')
cbar3 = plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
cbar3.set_label('Kurtosis', fontsize=8)
cbar3.ax.tick_params(labelsize=7)

for i in range(4):
    for j in range(5):
        idx = i * 5 + j
        axes[1, 1].text(j, i, f'{stats_data[idx, 3]:.2f}', 
                       ha='center', va='center', 
                       color='white' if abs(stats_data[idx, 3]) > 1 else 'black',
                       fontsize=7, fontweight='bold')

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.suptitle('Statistical metrics heatmaps', fontsize=11, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Figure3_statistical_metrics_heatmaps.pdf', dpi=600, bbox_inches='tight')
plt.close()

print("Saving Figure 4: Cumulative distribution functions...")
fig, ax = plt.subplots(figsize=(7.2, 5.4))

for i in range(20):
    temp_flat = temp_fracture[i].flatten()
    sorted_data = np.sort(temp_flat)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    ax.plot(sorted_data, cdf, color=nature_colors[i], linewidth=1.0, alpha=0.7)

ax.set_xlabel('Temperature (K)', fontsize=9)
ax.set_ylabel('Cumulative probability', fontsize=9)
ax.set_title('Cumulative distribution functions', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, linewidth=0.5)

legend_elements = [
    Line2D([0], [0], color=nature_colors[0], lw=1.5, label='Sample 1-5'),
    Line2D([0], [0], color=nature_colors[9], lw=1.5, label='Sample 6-15'),
    Line2D([0], [0], color=nature_colors[19], lw=1.5, label='Sample 16-20')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, framealpha=0.9, fancybox=False)

plt.tight_layout()
plt.savefig('Figure4_cumulative_distribution_functions.pdf', dpi=600, bbox_inches='tight')
plt.close()

print("Saving Figure 5: Percentile comparison...")
fig, ax = plt.subplots(figsize=(7.2, 5.4))

percentiles = [10, 25, 50, 75, 90]
percentile_data = np.zeros((20, len(percentiles)))

for i in range(20):
    temp_flat = temp_fracture[i].flatten()
    percentile_data[i] = np.percentile(temp_flat, percentiles)

x = np.arange(20)
width = 0.15

for j, p in enumerate(percentiles):
    offset = (j - (len(percentiles)-1)/2) * width
    color_idx = min(j * 4, 19)
    ax.bar(x + offset, percentile_data[:, j], width, 
           label=f'{p}th', color=nature_colors[color_idx], alpha=0.8,
           edgecolor='black', linewidth=0.5)

ax.set_xlabel('Sample index', fontsize=9)
ax.set_ylabel('Temperature (K)', fontsize=9)
ax.set_title('Percentile comparison', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{i+1}' for i in range(20)], rotation=45, fontsize=7)
ax.legend(title='Percentile', title_fontsize=8, fontsize=7, frameon=True, fancybox=False)
ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)

plt.tight_layout()
plt.savefig('Figure5_percentile_comparison.pdf', dpi=600, bbox_inches='tight')
plt.close()

print("Saving additional figures...")

fig, ax = plt.subplots(figsize=(7.2, 5.4))
for i in range(20):
    temp_flat = temp_fracture[i].flatten()
    sns.kdeplot(temp_flat, ax=ax, color=nature_colors[i], alpha=0.6, linewidth=1.0)

ax.set_xlabel('Temperature (K)', fontsize=9)
ax.set_ylabel('Probability density', fontsize=9)
ax.set_title('Temperature distribution comparison (KDE)', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.savefig('Figure6_kde_comparison.pdf', dpi=600, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(7.2, 5.4))
box = ax.boxplot(box_data, patch_artist=True, showfliers=False)

for i, patch in enumerate(box['boxes']):
    patch.set_facecolor(nature_colors[i])
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(0.5)

ax.set_xlabel('Sample index', fontsize=9)
ax.set_ylabel('Temperature (K)', fontsize=9)
ax.set_title('Box plot comparison', fontsize=10, fontweight='bold')
ax.set_xticklabels([f'{i+1}' for i in range(20)], rotation=45, fontsize=7)
ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)

plt.tight_layout()
plt.savefig('Figure7_boxplot_comparison.pdf', dpi=600, bbox_inches='tight')
plt.close()

print("\nSaving statistical summary...")
summary_data = []
for i in range(20):
    temp_flat = temp_fracture[i].flatten()
    summary_data.append([
        i+1,
        temp_flat.mean(),
        temp_flat.std(),
        temp_flat.min(),
        np.percentile(temp_flat, 25),
        np.median(temp_flat),
        np.percentile(temp_flat, 75),
        temp_flat.max(),
        calculate_skewness(temp_flat),
        calculate_kurtosis(temp_flat)
    ])

fig, ax = plt.subplots(figsize=(8.0, 10.0))
ax.axis('tight')
ax.axis('off')

table_data = []
headers = ['Sample', 'Mean', 'Std', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Skew', 'Kurt']

for row in summary_data:
    table_data.append([
        f'{row[0]:d}',
        f'{row[1]:.2f}',
        f'{row[2]:.2f}',
        f'{row[3]:.2f}',
        f'{row[4]:.2f}',
        f'{row[5]:.2f}',
        f'{row[6]:.2f}',
        f'{row[7]:.2f}',
        f'{row[8]:.3f}',
        f'{row[9]:.3f}'
    ])

all_temp = temp_fracture.flatten()
overall_stats = [
    'Overall',
    f'{all_temp.mean():.2f}',
    f'{all_temp.std():.2f}',
    f'{all_temp.min():.2f}',
    f'{np.percentile(all_temp, 25):.2f}',
    f'{np.median(all_temp):.2f}',
    f'{np.percentile(all_temp, 75):.2f}',
    f'{all_temp.max():.2f}',
    f'{calculate_skewness(all_temp):.3f}',
    f'{calculate_kurtosis(all_temp):.3f}'
]
table_data.append(overall_stats)

table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.2)

for j in range(len(headers)):
    table.auto_set_column_width(j)
    table[(0, j)].set_facecolor('#2E86AB')
    table[(0, j)].set_text_props(weight='bold', color='white')

for j in range(len(headers)):
    table[(len(table_data), j)].set_facecolor('#F4D35E')
    table[(len(table_data), j)].set_text_props(weight='bold')

plt.title('Statistical summary of temperature distributions\n20 samples × (128×128) data points each', 
          fontsize=12, fontweight='bold', pad=20)

plt.figtext(0.5, 0.02, 'Q1: 25th percentile, Q3: 75th percentile, Std: standard deviation', 
            ha='center', fontsize=7, style='italic')

plt.savefig('Statistical_Summary.pdf', dpi=600, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("FIGURES SAVED SEPARATELY:")
print("="*80)
print("1. Figure1_comprehensive_comparison.pdf - 4-panel comparison")
print("2. Figure2_detailed_distribution_analysis.pdf - 20 detailed histograms")
print("3. Figure3_statistical_metrics_heatmaps.pdf - Heatmaps of statistical metrics")
print("4. Figure4_cumulative_distribution_functions.pdf - CDF comparison")
print("5. Figure5_percentile_comparison.pdf - Percentile comparison")
print("6. Figure6_kde_comparison.pdf - Single KDE comparison")
print("7. Figure7_boxplot_comparison.pdf - Single box plot comparison")
print("8. Statistical_Summary.pdf - Complete statistical summary table")

print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)
print(f"{'Sample':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Q1':<10} "
      f"{'Median':<10} {'Q3':<10} {'Max':<10} {'Skew':<10} {'Kurt':<10}")
print("-"*110)

for row in summary_data:
    print(f"{row[0]:<8} {row[1]:<10.2f} {row[2]:<10.2f} {row[3]:<10.2f} "
          f"{row[4]:<10.2f} {row[5]:<10.2f} {row[6]:<10.2f} {row[7]:<10.2f} "
          f"{row[8]:<10.3f} {row[9]:<10.3f}")

print("\n" + "="*50)
print("KEY INFORMATION:")
print("="*50)
print(f"• Total samples: 20")
print(f"• Data points per sample: {128*128:,}")
print(f"• Total data points: {20*128*128:,}")
print(f"• Temperature range: {all_temp.min():.1f} - {all_temp.max():.1f} K")
print(f"• Overall mean ± s.d.: {all_temp.mean():.1f} ± {all_temp.std():.1f} K")
print(f"• Using provided color palette with {len(nature_colors)} colors")
print(f"• All figures saved in Nature journal style (600 DPI)")