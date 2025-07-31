import matplotlib.pyplot as plt
import numpy as np


def plot(fields, spectra,  distances, title=None):
    
    plt.style.use("fivethirtyeight")
    plt.rcParams.update(
        {
            "savefig.edgecolor": "white",
            "savefig.facecolor": "white",
            "figure.edgecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "white",
            "axes.facecolor": "white",
            "patch.edgecolor": "white",
            "patch.facecolor": "white",
            "patch.force_edgecolor": False,
        }
    )

    n_samples = len(fields)
    fig = plt.figure()#tight_layout=True)
    if title is not None:
        fig.suptitle(title, fontsize=14)
    
    fig, ax = plt.subplots(nrows=n_samples, ncols=2, figsize=(10, n_samples*3))
    # Field
    
    for j, field in enumerate(fields):
        shp = field.shape
        ax1 = ax[j, 0] if len(ax.shape) == 2 else ax[0]
        # ax1.axhline(y=0., color='k', linestyle='--', alpha=0.25) 
        im = ax1.imshow(field, origin="lower", cmap="RdBu_r")
        ax1.set_xticks(np.linspace(0, shp[0], 5), np.linspace(0, shp[0], 5) * distances[0])
        ax1.set_yticks(np.linspace(0, shp[1], 5), np.linspace(0, shp[1], 5) * distances[1])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Field realizations')
        fig.colorbar(im, ax=ax1)
    
    # Spectrum
        ax2 = ax[j, 1] if len(ax.shape) == 2 else ax[1]
        spectrum = spectra[j]
        xcoord = np.arange(len(spectrum))*1/distances[0] 
        spectrum = spectrum.at[0].set(spectrum[1])
        ax2.plot(xcoord, spectrum)
        ax2.set_ylim(1e-2, 1e6)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('k')
        ax2.set_ylabel('p(k)')
        ax2.set_title('Power Spectrum')
        
    fig.align_labels()
    plt.show()
    
def plot_nice(field, spectrum,  distances, path, name):
    
    plt.style.use("fivethirtyeight")
    plt.rcParams.update(
        {
            "savefig.edgecolor": "white",
            "savefig.facecolor": "white",
            "figure.edgecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "white",
            "axes.facecolor": "white",
            "patch.edgecolor": "white",
            "patch.facecolor": "white",
            "patch.force_edgecolor": False,
        }
    )

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    # Field
    
    # for j, field in enumerate(fields):
    minmax = 1.4 # 0.2*np.max(np.abs(field))
    shp = field.shape
    # ax1 = ax[0]
    ax1.grid(False)
    # ax1.axhline(y=0., color='k', linestyle='--', alpha=0.25) 
    im = ax1.imshow(field, origin="lower", cmap="RdBu_r", vmax=minmax, vmin=-minmax)
    #ax1.set_xticks(np.linspace(0, shp[0], 5), [f"{int(t):d}" for t in np.linspace(0, shp[0], 5) * distances[0]], fontsize=12)
    #ax1.set_yticks(np.linspace(0, shp[1], 5), [f"{int(t):d}" for t in np.linspace(0, shp[1], 5) * distances[1]], fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])

    #ax1.set_xlabel('x')
    #ax1.set_ylabel('y')
    ax1.yaxis.set_label_position("right")
    #ax1.set_title('Field realizations')
    cb = fig.colorbar(im, ax=ax1, location='left')
    
    cb.set_label(r"$v_x$")
    
    fig.tight_layout()
    # fig.align_labels()
    plt.savefig(path + name + "_field")
    plt.close()
    
    fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    vol = np.sqrt((field.shape[0]*distances[0])**2 + (field.shape[1]*distances[1])**2)
    
    # Spectrum
    # ax2 = ax[1]
    ax2.tick_params(axis='both', labelsize=12)
    xcoord = np.arange(len(spectrum))*1/distances[0] 
    spectrum = spectrum.at[0].set(spectrum[1])
    ax2.plot(xcoord, spectrum/vol*np.sqrt(2))
    ax2.set_ylim(1e-7, 1e3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel("k")
    ax2.set_ylabel("P(k)")
    #ax2.set_title('Power Spectrum')
    fig.tight_layout()
    # fig.align_labels()
    plt.savefig(path + name + "_power")
    
    plt.close()
    