#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 19:05:56 2026

@author: lkoehler
"""



import numpy as np
import subprocess
import h5py
import matplotlib.pyplot as plt
import json 
from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from scipy.signal import savgol_filter

import matplotlib.colors as mcolors
import math
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import trapezoid
import pickle
import matplotlib.ticker as ticker

import matplotlib.patches as ptc
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

import matplotlib.colors as colors

from scipy.spatial import Delaunay

from mpl_toolkits.axes_grid1 import make_axes_locatable


LABELFONTSIZE = 15
TICKSFONTSIZE = 12
LEGENDFONTSIZE = 12


local = False



name0, Nmax = "260421PhaseDiagramSoft", 960
name0, Nmax = "260428FromDense", 960


if local :
    directory_data = "/Users/lkoehler/Documents/Research/ViscoElasticActiveBrownianParticles/results/"
    
else:
    directory_data =  "/Users/lkoehler/Documents/PKSdata/ViscoElasticActiveBrownianParticles/"
    

directory_save = "/Users/lkoehler/Documents/Research/ViscoElasticActiveBrownianParticles/Results/"+name0

try:
    subprocess.run(['mkdir', directory_save])
except :
    donithing =0
    
#%% Load the data locally 

directory = "/Users/lkoehler/Documents/Research/ViscoElasticActiveBrownianParticles/results/"+name0+"/"
with open(directory+'all_data.pkl', 'rb') as handle:
    data = pickle.load(handle)
    
all_box_av = data["all_box_av"] 
all_box_std = data["all_box_std"] 
all_entropy = data["all_entropy"] 
all_msd = data["all_msd"] 
all_timelags_msd = data["all_timelags_msd"] 
all_steady_state_time = data["all_steady_state_time"] 
all_inflow_rate = data["all_inflow_rate"] 
all_orientation_velocity_correlation = data["all_orientation_velocity_correlation"] 
all_tau1 = data["all_tau1"] 
all_tau_propulsion = data["all_tau_propulsion"] 
all_tau_active = data["all_tau_active"] 
all_tau_elastic = data["all_tau_elastic"] 
all_Pe = data["all_Pe"] 
all_De = data["all_De"] 
all_gamma1 = data["all_gamma1"] 
all_Nparticles = data["all_Nparticles"] 
all_kT = data["all_kT"] 
all_dt_saved = data["all_dt_saved"] 
all_diameters = data["all_diameters"] 
all_v0 = data["all_v0"] 
all_alpha = data["all_alpha"] 
all_packing_fraction = data["all_packing_fraction"] 
all_delta = data["all_delta"] 
all_cluster_fraction_mean = data["all_cluster_fraction_mean"]
all_cluster_fraction_std = data["all_cluster_fraction_std"]
all_cluster_fraction_evolution = data["all_cluster_fraction_evolution"]




index_to_PeDe = {}
PeDe_to_index = {}
    

for i in all_Pe.keys():
            
    
        Peclet = all_Pe[i]
        Deborah = all_De[i]
    
        key = (Peclet, Deborah)
        if key in PeDe_to_index.keys():
            PeDe_to_index[key].append(i)
        else:
            PeDe_to_index[key] = [i]
        index_to_PeDe[i] = key
          
        
        
# Set the colormap 
VMIN = min(all_cluster_fraction_mean.values())
VMAX = max(all_cluster_fraction_mean.values())
        
MIPS_THRESHOLD = 0.3

cmap = plt.get_cmap("coolwarm")  # or your current cmap

norm = mcolors.TwoSlopeNorm(
    vmin=VMIN,
    vcenter=MIPS_THRESHOLD,
    vmax=VMAX
)


        #%%
def plot_state(t, positions, orientation, radius, ax=None, plot_text=True, plot_title=True, title=None, Lx=1, Ly=1):
    """Plot particles with the color refering to the phase and the correct size"""
    
    xs = (positions[0,:,t])%Lx
    ys = (positions[1,:,t])%Ly
    thetas = orientations[0,:,t]
    thetas = (thetas)%(2*np.pi)
    
    # with open(directory_data + name +"/ParametersSimulation.json", "r") as f:
    #      parameters_simulation = json.load(f)
    # with open(directory_data + name +"/ParametersEvolution.json", "r") as f:
    #      parameters_evolution = json.load(f)
    # with open(directory_data + name +"/ParametersInitialisation.json", "r") as f:
    #      parameters_initialisation = json.load(f)
    
    # Circle positions and radii
    centers = np.column_stack((xs, ys))  # along x-axis
    radii = radius  * np.ones(len(xs))                         # increasing radius

    # Sequential colors from a colormap
    cmap = cm.get_cmap("hsv")
    colors = cmap(thetas/(2*np.pi))  # N colors from viridis colormap

    # Create circle patches
    circles = [Circle((cx, cy), r) for (cx, cy), r in zip(centers, radii)]

    # Vectorized rendering with PatchCollection
    collection = PatchCollection(circles, facecolor=colors, edgecolor="black", alpha=0.8, linewidth=0)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
    ax.add_collection(collection)
    # ax.scatter(xs, ys, c=Pfull, cmap=cmap, s=1, vmin=0, vmax=1)
    # ax.scatter(Xfull, Yfull, colors=Pfull, s=1, vmin=0, vmax=1)
    #ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([])
    ax.set_yticks([])
    if plot_text:
        ax.text(0.93,0.01,str(t),fontsize=20)
    
    if plot_title:
        plt.title(title, fontsize=17)
    plt.tight_layout()
    
    
    return(ax)


def compute_pbc_delaunay_neighbors(x, y, box_length=1.0):
    # first, recenter the points in the box to [0, box_length)
    x_ctr = np.mod(x, box_length)
    y_ctr = np.mod(y, box_length)
    N = len(x)
    
    # replicate points in neighboring boxes for Delaunay triangulation
    pts = np.stack([x_ctr, y_ctr], axis=1)
    for xx in np.arange(-1, 2):
        for yy in np.arange(-1, 2):
            if xx == 0 and yy == 0:
                continue
            pts = np.append(pts, np.stack([x_ctr + xx * box_length, y_ctr + yy * box_length], axis=1), axis=0)
    tri_orig = Delaunay(pts)
    
    # loop over the neighbors of the first N points (the original points). If a neighbor is in
    # the original set, add to the list of neighbors for that particle. If not, find the 
    # image in the middle box and add that as a neighbor.
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        for j in tri_orig.vertex_neighbor_vertices[1][tri_orig.vertex_neighbor_vertices[0][i]:tri_orig.vertex_neighbor_vertices[0][i+1]]:
            if j < N:
                neighbors[i].append(j)
            else:
                # find the image in the middle box
                j_mod = j % N
                neighbors[i].append(j_mod)

    
    return neighbors


def find_clusters(x, y, sigma, box_length=1.0):
    """
    Find connected clusters of particles using Newman-Ziff algorithm.
    
    Parameters:
        x, y: 1D arrays of particle positions
        sigma: distance threshold for connectivity
        box_length: side length of the periodic box (assume square, default 1.0)
    
    Returns:
        labels: array of cluster labels for each particle
        cluster_sizes: array of cluster sizes (indexed by cluster label)
    """
    N = len(x)
    
    # center positions in box
    x_ctr = np.mod(x, box_length)
    y_ctr = np.mod(y, box_length)
    
    # Union-Find data structure
    parent = np.arange(N, dtype=int)
    rank = np.zeros(N, dtype=int)
    
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]  # path compression
            a = parent[a]
        return a
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
    
    # Use Delaunay triangulation to avoid O(N^2) brute force (optional optimization)
    neighbors = compute_pbc_delaunay_neighbors(x_ctr, y_ctr, box_length)
    
    # loop over neighbors and union if within distance
    for i in range(N):
        for j in neighbors[i]:
            if j > i:  # avoid double counting
                dx = x_ctr[j] - x_ctr[i]
                dy = y_ctr[j] - y_ctr[i]
                # Minimum image convention
                dx -= box_length * np.round(dx / box_length)
                dy -= box_length * np.round(dy / box_length)
                dist = np.sqrt(dx**2 + dy**2)
                if dist < sigma:
                    union(i, j)
    
    # Assign cluster labels
    root_to_label = {}
    labels = np.zeros(N, dtype=int)
    current_label = 0
    for i in range(N):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = current_label
            current_label += 1
        labels[i] = root_to_label[root]
    
    # Compute cluster sizes
    num_clusters = current_label
    cluster_sizes = np.zeros(num_clusters, dtype=int)
    for i in range(N):
        cluster_sizes[labels[i]] += 1
    
    return labels, cluster_sizes





def load_simulation_particle_images(
    index,
    name0,
    directory_data,
    image_dict, 
    frame_to_plot="last",
    box_length=1.0,
    cluster_sigma_factor=0.98,
    verbose=False,
):

    name = name0 + str(index)

    data_file = directory_data + "/" + name + "/all_data.h5"
    param_evol =directory_data + "/" + name + "/ParametersEvolution.json"
    param_sim = directory_data + "/" + name + "/ParametersSimulation.json"

    with h5py.File(data_file, "r") as f:
        positions = f["positions"][:]

        if verbose:
            for key in f.keys():
                print(f"{key}: {f[key].shape}")

    with open(param_evol, "r") as f:
        param_evol_data = json.load(f)

    with open(param_sim, "r") as f:
        param_sim_data = json.load(f)

    PARTICLE_SIZE = param_evol_data["PARTICLE_SIZE"]
    SAVING_PERIOD = param_sim_data["SAVING_PERIOD"]
    DT = param_sim_data["DT"]

    v0 = param_evol_data["PARTICLE_SELF_PROPULSION"]
    Dtheta = param_evol_data["NOISE_STRENGTH_ORIENTAITON"]
    sigma = param_evol_data["PARTICLE_SIZE"]

    Peclet = np.round(v0 / sigma / Dtheta, 3)

    NFRAMES = positions.shape[-1]

    if frame_to_plot == "last":
        FRAME_2_PLOT = NFRAMES - 1
    else:
        FRAME_2_PLOT = int(frame_to_plot)

    x = positions[0, :, FRAME_2_PLOT]
    y = positions[1, :, FRAME_2_PLOT]

    labels, cluster_sizes = find_clusters(
        x,
        y,
        sigma=PARTICLE_SIZE * cluster_sigma_factor,
        box_length=box_length,
    )

    max_cluster_fraction = np.max(cluster_sizes) / positions.shape[1]

    MAX_CLUSTER_SIZE = positions.shape[1]

    particle_cluster_sizes = cluster_sizes[labels]

    x_ctr = np.mod(x, box_length)
    y_ctr = np.mod(y, box_length)

    image_dict[index] = {
        "x": x_ctr,
        "y": y_ctr,
        "particle_cluster_sizes": particle_cluster_sizes,
        "particle_size": PARTICLE_SIZE,
        "peclet": Peclet,
        "max_cluster_fraction": max_cluster_fraction,
        "frame": FRAME_2_PLOT,
        "nframes": NFRAMES,
        "saving_period": SAVING_PERIOD,
        "dt": DT,
        "max_cluster_size": MAX_CLUSTER_SIZE,
    }

    return image_dict

        
        
        


def plot_simulation_particle_image(
    image_dict,
    index,
    ax=None,
    show_colorbar=False,
    fig=None,
    edgecolor="black",
    linewidth=0.25,
    alpha=0.8,
    cmap_name = "coolwarm"
):
    """
    Plot one simulation image into a given axis.

    Parameters
    ----------
    image_dict : dict
        Output from load_simulation_particle_images.
    index : int
        Simulation index to plot.
    ax : matplotlib axis, optional
        Axis to plot into. If None, creates a new figure and axis.
    show_colorbar : bool
        Whether to add a cluster-size colorbar.
    fig : matplotlib figure, optional
        Needed only if show_colorbar=True and ax belongs to a bigger figure.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    elif fig is None:
        fig = ax.figure

    data = image_dict[index]

    x = data["x"]
    y = data["y"]
    particle_cluster_sizes = data["particle_cluster_sizes"]

    cmap = plt.get_cmap(cmap_name)
    
    
    vmin = VMIN * data["max_cluster_size"]
    vmax = VMAX * data["max_cluster_size"]
    vcenter = MIPS_THRESHOLD * data["max_cluster_size"]

    norm = mcolors.TwoSlopeNorm(
        vmin=vmin,
        vcenter=vcenter,
        vmax=vmax,
    )

    colors = cmap(norm(particle_cluster_sizes))

    # norm = mcolors.Normalize(
    #     vmin=0.05*data["max_cluster_size"],
    #     vmax=0.7*data["max_cluster_size"]
    # )
    
    # colors = cmap(norm(particle_cluster_sizes))

    particle_size = data["particle_size"]

    circles = [
        ptc.Circle(
            (x[j], y[j]),
            radius=(0.95 * particle_size) / 2,
        )
        for j in range(len(x))
    ]

    collection = PatchCollection(
        circles,
        facecolor=colors,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )

    ax.add_collection(collection)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if show_colorbar:
        sm = plt.cm.ScalarMappable(
            cmap=data["cmap"],
            norm=data["norm"],
        )
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Cluster size")
        cbar.ax.tick_params(labelsize=12)

    return ax




        
        

#%% Phase diagram of the cluster fraction


DELTA0 = 0.05

tau_list = np.sort(np.unique(np.array(list(all_tau1.values()))))#[0:-3]
tau_ratio_list = np.sort(np.unique(np.array(list(all_De.values())))) #[:-4]#[1:]

Pe_list = np.sort(np.unique(np.array(list(all_Pe.values()))))[1:]

# Pe_list = [ 5, 10,15,20,25, 30]

cluster_matrix = np.full((len(Pe_list), len(tau_ratio_list)), np.nan)
N_mes_matrix = np.full((len(Pe_list), len(tau_ratio_list)), np.nan)
N, M = cluster_matrix.shape
De_index = {v: i for i, v in enumerate(tau_ratio_list)}
Pe_index = {v: i for i, v in enumerate(Pe_list)}



# MES_TIME = 20
MES_DURATION = 10
MES_TIME = 200-MES_DURATION

evo_save = []
# for De, Pe, cluster, cluster_evo, tau_act, delta, phi in zip(all_De.values(), all_Pe.values(), all_cluster_fraction_mean.values(), all_cluster_fraction_evolution.values(), all_tau_active.values(), all_delta.values(), all_packing_fraction.values()):
for index in all_cluster_fraction_evolution.keys():
    De = all_De[index]
    Pe = all_Pe[index]
    cluster_evo = all_cluster_fraction_evolution[index]
    cluster = all_cluster_fraction_mean[index]

        
    # tmin, tmax = MES_TIME, MES_TIME + MES_DURATION
    # if tmin > len(cluster_evo):
    #     tmin =  max(0,len(cluster_evo) - MES_DURATION)
    #     tmax = len(cluster_evo)
    
    # cluster = np.nanmean(cluster_evo[tmin:tmax])
    # if De==0 and Pe==17.5:
    #     evo_save.append(cluster_evo)
    #     print(tmin, tmax, cluster )
    try:
        i = Pe_index[Pe]
        j = De_index[De]#tau_index[tau/tau_act]
        if i<N and j<M:
            if np.isnan(cluster_matrix[i, j]):
                
                cluster_matrix[i, j] = cluster
                N_mes_matrix[i, j] = 1
            else:
                cluster_matrix[i, j] += cluster
                N_mes_matrix[i, j] += 1
    except: 
        donothing=0

# example arrays


cluster_matrix = cluster_matrix/N_mes_matrix

De0_column = cluster_matrix[:,0]
cluster_matrix = cluster_matrix[:,1:]

T, P = np.meshgrid(tau_ratio_list[1:], Pe_list)


vmax = np.max(list(all_cluster_fraction_mean.values()))


# plt.figure(figsize=(6,5))


fig, ax = plt.subplots(figsize=(7,6))

# # shared normalization so both panels use same colorbar
# norm = colors.Normalize(
#     vmin=np.nanmin(cluster_matrix),
#     vmax=np.nanmax(cluster_matrix)
# )




# ----------------------------
# Plot phase diagram log scale 
# ----------------------------

T_centers = T[0]
logT = np.log10(T_centers)
logT_edges = np.empty(len(T_centers) + 1)
logT_edges[1:-1] = 0.5 * (logT[:-1] + logT[1:])
logT_edges[0] = logT[0] - 0.5 * (logT[1] - logT[0])
logT_edges[-1] = logT[-1] + 0.5 * (logT[-1] - logT[-2])
T_edges = 10**logT_edges
P_centers = P[:, 0]
P_edges = np.empty(len(P_centers) + 1)
P_edges[1:-1] = 0.5 * (P_centers[:-1] + P_centers[1:])
P_edges[0] = P_centers[0] - 0.5 * (P_centers[1] - P_centers[0])
P_edges[-1] = P_centers[-1] + 0.5 * (P_centers[-1] - P_centers[-2])



mesh = ax.pcolormesh(
    T_edges, P_edges, cluster_matrix,
    shading="flat",
    cmap="coolwarm",
    norm=norm,
    edgecolors="none",
    linewidth=0,
    antialiased=False,
)

ax.set_xscale("log")
ax.set_xlabel(r"Memory timescale $\tau_\mathrm{m}/\tau_\mathrm{a}$", fontsize=LABELFONTSIZE)
ax.set_ylabel(r"Peclet number $\mathrm{Pe}$", fontsize=LABELFONTSIZE)
ax.tick_params(axis="both", labelsize=TICKSFONTSIZE)


xlim = plt.xlim()
ylim = plt.ylim()



# ax.set_yticks([])

#----------------------------
# ---- zero-column panel ----
#----------------------------
# data for the extra column, shape should be (len(P_centers),)
zero_col = De0_column

divider = make_axes_locatable(ax)

# small axis on the left, with tiny gap
ax0 = divider.append_axes("left", size="6%", pad=0.04, sharey=ax)

# fake x edges around 0, so it appears as one column centered at x=0
x0_edges = [-0.5, 0.5]

mesh0 = ax0.pcolormesh(
    x0_edges,
    P_edges,
    zero_col[:, None],
    shading="flat",
    cmap="coolwarm",
    norm=norm,
    edgecolors="none",
    linewidth=0,
    antialiased=False,
)

ax0.set_xlim(-0.5, 0.5)
ax0.set_xticks([0])
ax0.set_xticklabels(["0"])
ax0.tick_params(axis="both", labelsize=TICKSFONTSIZE)


# hide y ticks/labels on main plot
ax.tick_params(axis='y', left=False, labelleft=False)

# keep them on the left mini-axis
ax0.tick_params(axis='y', left=True, labelleft=True)



# ----------------------------
#  Legends and labels 
# ----------------------------
# Put main y label only on the zero panel, since it is now leftmost
ax.set_ylabel("")
ax0.set_ylabel(r"Peclet number $\mathrm{Pe}$", fontsize=LABELFONTSIZE)

# colorbar shared by both, using same norm
cbar = fig.colorbar(mesh, ax=[ax0, ax])
cbar.set_label(r"Fraction in biggest cluster", fontsize=LABELFONTSIZE)
cbar.ax.tick_params(labelsize=TICKSFONTSIZE)

# add horizontal line
cbar.ax.axhline(
    MIPS_THRESHOLD,
    color="black",
    linewidth=1,
)



# ----------------------------------------------------------------------------
# Replot other contours 
# ----------------------------------------------------------------------------

legends = ["Free particle theory", "Theoretical transition", "Collision correction"]
color_contour = [ "black", "black","black"]
linestyle_contour = ["-", "--", "-."]
linewidth_contour = [1.5, 1.5, 1]
level_contour = [0.5, 0.5, 0.35]
names_contour = ["theory_1p", "theory"]#, "theory_collision"]#, "260428FromDense"]

handles = []
for icontour, name_contour in enumerate(names_contour):
    print(name_contour)
    directory_contour_data = "/Users/lkoehler/Documents/Research/ViscoElasticActiveBrownianParticles/local/paper/PhaseDiagram/"+name_contour
    with open(directory_contour_data+'_contour_data.pkl', 'rb') as handle:
        contour_data_read = pickle.load(handle)
        
    TT, PP, Zpad = contour_data_read["TT"], contour_data_read["PP"], contour_data_read["Zpad"]
    ax.contour( TT,PP,Zpad, levels=[level_contour[icontour]],colors=color_contour[icontour], 
                linewidths=linewidth_contour[icontour], linestyles=linestyle_contour[icontour]
                )
    handles.append(
    Line2D(
        [0], [0],
        color=color_contour[icontour],
        linestyle=linestyle_contour[icontour],
        linewidth=linewidth_contour[icontour],
        label=legends[icontour],
    )
)
ax.legend(handles=handles, loc="lower left", framealpha=0.5, fontsize=LEGENDFONTSIZE)


ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_ylim(2.5, 52.5)
mesh.set_rasterized(True)


plt.savefig("/Users/lkoehler/Documents/Research/ViscoElasticActiveBrownianParticles/local/paper/PhaseDiagram/PhaseDiagram.pdf", bbox_inches = "tight", transparent=True)


        
        
#%% Panels of images - preload the data 



directory = "/Users/lkoehler/Documents/Research/ViscoElasticActiveBrownianParticles/local/paper/PhaseDiagram/"
with open(directory+name0+'_img_data.pkl', 'rb') as handle:
    saved_img_data = pickle.load(handle)
    
    

Pe_list = [12.5, 17.5, 22.5]#, 27.5, 35]
De_list = [0.001, 0.1, 1, 5]

for Pe in Pe_list:
    for De in De_list:
        indices = PeDe_to_index[(Pe, De)]
        
        index = indices[np.argmax([all_cluster_fraction_mean[i] for i in indices])]
        
        print(index)
        if index not in saved_img_data.keys():
        
            load_simulation_particle_images(index, name0, directory_data, saved_img_data)
    
directory = "/Users/lkoehler/Documents/Research/ViscoElasticActiveBrownianParticles/local/paper/PhaseDiagram/"
with open(directory+name0+'_img_data.pkl', 'wb') as handle:
    pickle.dump(saved_img_data, handle)
    
        
        
        
        
        
#%% Panels of images, do the plot 


im_size = 2

f, axes = plt.subplots(3, 4, figsize=(im_size*4, im_size*3))

plt.subplots_adjust(wspace=0.02, hspace=0.02)

Pe_list = [12.5, 17.5, 22.5]
De_list = [0.001, 0.1, 1, 5]

for iPe, Pe in enumerate(Pe_list):
    for iDe, De in enumerate(De_list):
        indices = PeDe_to_index[(Pe, De)]
        index = indices[np.argmax([all_cluster_fraction_mean[i] for i in indices])]
        
        print(Pe, De, index, all_cluster_fraction_mean[index])
        
        ax = axes[3-iPe-1, iDe]
        plot_simulation_particle_image(
            saved_img_data,
            index,
            ax=ax,
            show_colorbar=False,
            fig=None,
            edgecolor="black",
            linewidth=0.0,
            alpha=1,
        )
        if De >= 1e-2:
            De_str = f"{De:g}"
        else:
            exponent = int(np.log10(De))
            De_str = rf"10^{{{exponent}}}"
            
        ax.text(0.5, 0.9, 
                r"$\mathrm{Pe}="+str(Pe)+r"$, $\tau_\mathrm{m}/\tau_\mathrm{a}="+De_str+"$",
                ha="center",
                va="center",
                fontsize=LEGENDFONTSIZE-3,
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="dimgray",
                    boxstyle="round,pad=0.3"))
    
        
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)


f.text(0.25, 0.04, r"Memory timescale $\tau_\mathrm{m}/\tau_\mathrm{a}$", fontsize=LABELFONTSIZE)
f.text(0.09, 0.27, r"Peclet number $\mathrm{Pe}$", rotation=90, fontsize=LABELFONTSIZE)

plt.savefig("/Users/lkoehler/Documents/Research/ViscoElasticActiveBrownianParticles/local/paper/PhaseDiagram/PhaseDiagram_panels.png", bbox_inches = "tight", transparent=True, dpi=200)




#%% Panels of v_eff and tau_eff

def two_exp_model(t, A, tau1, B, C, tau2):
    return A*np.exp(-t/tau1) + (B )*np.exp(-t/tau2)


def three_exp_model(t, A, tau1, B, tau2, C, tau3):
    return A*np.exp(-t/tau1) + (B )*np.exp(-t/tau2) + (C )*np.exp(-t/tau3)


def C_vn_t(t, gamma0, gamma1, tau1, Dtheta, v0):
    t = np.asarray(t, dtype=float)
    gamma_hat = gamma0 + gamma1
    tau_a = 1.0 / Dtheta
    tau_v = gamma0 * tau1 / gamma_hat

    # ── Limit tau1 → 0: Markovian limit → v0·exp(-t/τ_a) ────────────────────
    if np.isclose(tau1, 0.0):
        return v0 * np.exp(-t / tau_a)

    # ── Degenerate: tau_v ≈ tau_a ────────────────────────────────────────────
    if np.isclose(tau_v, tau_a):
        return gamma_hat * v0 * (
              (1.0 / gamma0) * np.exp(-t / tau_a)
            - (gamma1 / (gamma0**2 * tau1)) * (t + tau_a / 2.0) * np.exp(-t / tau_a)
        )

    # ── Generic ──────────────────────────────────────────────────────────────
    integral = (tau_a * tau_v / (tau_a + tau_v)) * (
        (tau_a + tau_v) * np.exp(-t / tau_a) - 2.0 * tau_v * np.exp(-t / tau_v)
    ) / (tau_a - tau_v)

    return gamma_hat * v0 * (
          (1.0 / gamma0) * np.exp(-t / tau_a)
        - (gamma1 / (gamma0**2 * tau1)) * integral
    )




# -----------------------------------------
# Calculate the theory 1 particle curves 
# -----------------------------------------


De_list_theory = np.arange(0.005, 5, 0.005)
tau_eff_theory = []
v_eff_theory = []

tau_a = 200
v0 = 10
times = np.arange(0, 10*tau_a, tau_a/100)
for De in De_list_theory:
    Ct = C_vn_t(times, 0.1, 0.9, De*tau_a, 1/tau_a, v0)
    yt = Ct / Ct[0]
    tau_eff_theory.append(np.trapezoid(np.abs(yt), times))
    v_eff_theory.append(Ct[0])                      # ← fix 1: correct indentation

tau_eff_theory  = np.array(tau_eff_theory)/tau_a   # shape (Nt,)
v_eff_theory  = np.array(v_eff_theory) /v0   # shape (Nt,)



Pe_list = [5,10, 15,  20, 25]
# Pe_list = [5, 10]#, 15, 20, 30]
Pe_max = max(Pe_list)+10
De_list = np.sort(np.unique(np.array(list(all_De.values()))))[:]
De_list = [0, 0.1, 0.2,  0.5,0.7, 0.9, 1, 1.5, 2, 3, 4, 5]#, 2, 3]
# De_list = [0,  0.5, 1]

plot_correl = False

all_tau_eff = {}
all_l_eff = {}
all_v_eff = {}

if plot_correl :
    plt.figure()

for iPe, Pe_chosen in enumerate(Pe_list) :
    
    tau_eff_list = np.zeros_like(De_list, dtype=float)
    l_eff_list = np.zeros_like(De_list, dtype=float)
    n_mes_list = np.zeros_like(De_list)
    v_eff_list = np.zeros_like(De_list, dtype=float)

    for iDe, De in enumerate(De_list):
        
        
        if (Pe_chosen, De) in PeDe_to_index.keys():
            indices = PeDe_to_index[(Pe_chosen, De)]
    
        
            for index in indices[-1:] :
                
                times = np.asarray(all_timelags_msd[index])
                Cvn = np.asarray(all_orientation_velocity_correlation[index])
                
                
                
                    
                
                # if np.sum(Cvn)!=0:
                if True:
                    npoints = min(len(times), len(Cvn))
                    times = times[:npoints]
                    Cvn = Cvn[:npoints]
                
                    τm = all_tau1[index]
                    label = f"τm/τa={all_tau1[index]/all_tau_active[index]:.2g}, Pe={Pe_chosen}"
                    τactive = all_tau_active[index]
                
                
                    if plot_correl:
                        plt.plot(times, Cvn, "o", ms=3, alpha=0.9, label=label)
                
                    # fit only finite data
                    mask = np.isfinite(times) & np.isfinite(Cvn) 
                    t_clean = times[mask]
                    Cvn_clean = Cvn[mask]
                    
                
                    if len(t_clean) < 6:
                        continue
                
                
                    if De!=0:
                        γ1 = 0.1
                    else:
                        γ1 = 0.0
                    γx = 1-γ1
                    v0 = all_v0[index]
                    
                    tau_v = γx*τm / (γx+γ1)
                    γ0 = γx
                    An_guess = (γ0+γ1)*v0 * (1/γ0 - (γ1 * τactive * tau_v)/ (γ0**2 * τm*(τactive - tau_v) ) )
                    Bn_guess = (γ0+γ1)*v0 * ((2* γ1 * τactive * tau_v**2)/ (γ0**2 * τm*(τactive**2 - tau_v**2) ) )
                
                    
                    
                    # -------------------------------
                    # Fit with three exponentials
                    # -------------------------------
                    p0 = [
                        0.5*Cvn_clean[0], t_clean.max()/50,
                        0.3*Cvn_clean[0], t_clean.max()/10,
                        0.2*Cvn_clean[0], t_clean.max()/2
                    ]
                    bounds = (
                        [-np.inf, 1e-12, -np.inf, 1e-12, -np.inf,1e-12],
                        [ np.inf, np.inf, np.inf,  np.inf, np.inf, np.inf]
                    )
                    popt, pcov = curve_fit(
                        three_exp_model,
                        t_clean,
                        Cvn_clean,
                        p0=p0,
                        bounds=bounds,
                        maxfev=40000
                    )
                
                    A, tau1, B, tau2, C, tau3 = popt
                    
                    dt_fit = 0.05
                    t_fit = np.arange(0.01, τactive*20, dt_fit)
                    C_fit3 = three_exp_model(t_fit, *popt)
                    v_eff = C_fit3[0]
                    tau_eff = np.sum(np.abs(C_fit3))/C_fit3[0] *dt_fit
                    l_eff = np.sum(np.abs(C_fit3)) *dt_fit
                    if C_fit3[0] == 0:
                        l_eff = np.nan
            
                    else:
                    
                        tau_eff_list[iDe] += tau_eff / τactive
                        l_eff_list[iDe] += l_eff / (v0*τactive)
                        v_eff_list[iDe] += v_eff / v0
                        n_mes_list[iDe] += 1
                        
                        
                        if plot_correl:
                            plt.plot(t_fit, C_fit3, lw=1, color="grey", label=f"τeff/τa={tau_eff / τactive:.2g}")
                            
                    
                    
                # else:
                    # tau_eff_list[iDe] = np.nan
                    # l_eff_list[iDe] = np.nan
                
        
    tau_eff_list = np.where(n_mes_list>0, tau_eff_list, np.nan)
    l_eff_list = np.where(n_mes_list>0, l_eff_list, np.nan)
    v_eff_list = np.where(n_mes_list>0, v_eff_list, np.nan)
    # print(n_mes_list)
    n_mes_list = np.where(n_mes_list>0, n_mes_list, 1)
    all_tau_eff[Pe_chosen] = tau_eff_list/n_mes_list.copy()
    all_l_eff[Pe_chosen] = l_eff_list/n_mes_list.copy()
    all_v_eff[Pe_chosen] = v_eff_list/n_mes_list.copy()
        
if plot_correl:
    plt.legend(loc="upper right")
    plt.xlim(-1, 2*τactive)
    plt.ylabel("orientation velocity correlation")
    plt.xlabel("Δt")
    plt.title(f"τa={τactive}")


    


markers = ["o", "s", "d", "^", "*"]
if not plot_correl:
    f, axes = plt.subplots(2, 1, figsize=(3.5,5.5))
    
    # Plot the theory 
    axes[0].plot(De_list_theory, tau_eff_theory, color="black")
    axes[1].plot(De_list_theory, v_eff_theory, color="black", label="Free particle theory")
    
    axes[1].plot(De_list, 0.5+np.sqrt(De_list), label=r"$(1-\phi)(1+2\sqrt{\tau_m/\tau_a})$", color="black", linestyle="--")
    
    
    for iPe, Pe in enumerate(Pe_list):
        
        nPe = len(Pe_list)+2
        axes[0].scatter(De_list, all_tau_eff[Pe], marker=markers[iPe], facecolor="none", edgecolor=cm.Greens((nPe-iPe)/nPe), label=f"Pe={Pe}",)
        # axes[1].plot(De_list, all_l_eff[Pe], label=f"Pe={Pe}", marker=".")
        axes[1].scatter(De_list, all_v_eff[Pe],  marker=markers[iPe], facecolor="none", edgecolor=cm.Greens((nPe-iPe)/nPe))
    # axes[1].plot(De_list,2 *(0.5+np.sqrt(De_list)), label=r"$\frac{1}{2}+\sqrt{\tau_m/\tau_a}$", color="darkgrey", linestyle="--")
    
    # axes[1].legend(fontsize=15)
    axes[0].legend(fontsize=LEGENDFONTSIZE)#loc="upper right")
    axes[1].legend(fontsize=LEGENDFONTSIZE, loc="upper left")
    
    
    
    # axes[0].set_xlabel(r"Memory timescale $\tau_m/\tau_a$", fontsize=15)
    axes[1].set_xlabel(r"Memory timescale $\tau_m/\tau_a$", fontsize=LABELFONTSIZE)
    
    # axes[0].set_xticks([])
    axes[0].tick_params(axis='x', labelbottom=False)
    axes[1].set_yticks(np.arange(0.5, 4.1, 0.5))
    
    axes[0].set_ylabel(r"Eff. persistence time $\tau_{\mathrm{eff}}/\tau_\mathrm{a}$", fontsize=LABELFONTSIZE)
    axes[1].set_ylabel(r"Effective velocity $v_{\mathrm{eff}}/v_0$", fontsize=LABELFONTSIZE)
    
    
    axes[0].set_ylabel(r"$\tau_{\mathrm{eff}}/\tau_\mathrm{a}$", fontsize=LABELFONTSIZE)
    axes[1].set_ylabel(r"$v_{\mathrm{eff}}/v_0$", fontsize=LABELFONTSIZE)

    
    for i in range(2):  
        axes[i].tick_params(axis='both', labelsize=TICKSFONTSIZE)
        
     
    plt.tight_layout()
    # plt.subplots_adjust(hspace=0.15)
    
    plt.savefig("/Users/lkoehler/Documents/Research/ViscoElasticActiveBrownianParticles/local/paper/PhaseDiagram/VeffTauEff.pdf", bbox_inches = "tight", transparent=True)




#%% Initial condition 

name = name0+str(9)
title = "N="+str(all_Nparticles[index])+", τ1="+str(all_tau1[index])+", Pe="+str(all_Pe[index])
filename = directory_data + name + "/all_data.h5"
with h5py.File(filename, "r") as f:
    positions = f["positions"][:]      # loads only this dataset
with open(directory_data + name +"/ParametersEvolution.json", "r") as f:
     parameters_evolution = json.load(f)
with open(directory_data + name +"/ParametersSimulation.json", "r") as f:
     parameters_simulation = json.load(f)
#%%
t= 0
Lx, Ly = 1, 1

xs = (positions[0,:,t])%Lx
ys = (positions[1,:,t])%Ly
xs, ys = ys, xs

xs = (xs + 0.04)%1
radius = parameters_evolution["PARTICLE_SIZE"]/2

# Circle positions and radii
centers = np.column_stack((xs, ys))  # along x-axis
radii = radius  * np.ones(len(xs))                         # increasing radius

colors = cm.coolwarm(0.99)

# Create circle patches
circles = [Circle((cx, cy), r) for (cx, cy), r in zip(centers, radii)]

# Vectorized rendering with PatchCollection
collection = PatchCollection(circles, facecolor=colors, edgecolor="black", alpha=1, linewidth=0)

fig, ax = plt.subplots(figsize=(2.7,2.7))
fig.patch.set_alpha(0)
ax.set_facecolor((1, 1, 1, 0.9))
ax.add_collection(collection)
# ax.scatter(xs, ys, c=Pfull, cmap=cmap, s=1, vmin=0, vmax=1)
# ax.scatter(Xfull, Yfull, colors=Pfull, s=1, vmin=0, vmax=1)
#ax.set_aspect("equal", adjustable="box")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xticks([])
ax.set_yticks([])

ax.text(0.5, 0.9, "Initial condition", fontsize=LABELFONTSIZE, ha="center")

plt.savefig("/Users/lkoehler/Documents/Research/ViscoElasticActiveBrownianParticles/local/paper/PhaseDiagram/InitialCondition.pdf", bbox_inches = "tight")#, transparent=True)





        
        
        
        
        
        