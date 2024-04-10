import csv
import gzip
import os
import scipy.io
import pandas as pd
# import h5py as h5

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import timeit
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture

from collections import Counter

from scipy.stats import entropy

import seaborn as sns

from sklearn.manifold import TSNE, Isomap, SpectralEmbedding

from pointpats import PointPattern, PoissonPointProcess, as_window
from pointpats import g_test,f_test,k_test,j_test,l_test#G, F, J, K, L, Genv, Fenv, Jenv, Kenv, Lenv
#%matplotlib inline


np.random.seed(29)



def main(clustering_method, n_clusters, ori_img_path, matrix_path, features_path, barcodes_path, spatial_list_path):
    # matrix_dir = "filtered_feature_bc_matrix"
    mat = scipy.io.mmread(matrix_path).toarray()
    data = mat.T

    X_reduced = PCA(n_components=0.98).fit_transform(data)

    labels = clustering(X_reduced, clustering_method, n_clusters)

    # features_path = os.path.join(matrix_dir, "features.tsv.gz")
    feature = pd.read_csv(gzip.open(features_path), delimiter="\t", header=None).values
    feature_ids = [row[0] for row in feature]
    gene_names = [row[1] for row in feature]
    feature_types = [row[2] for row in feature]
    # barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
    barcodes = pd.read_csv(gzip.open(barcodes_path), delimiter="\t", header=None).values


    # spatial_list_path = r"filtered_feature_bc_matrix\spatial\tissue_positions_list.csv"
    spatial_info = pd.read_csv(spatial_list_path, delimiter=",", header=None).values


    # hires_img_path = r"filtered_feature_bc_matrix\spatial\tissue_hires_image.png"
    # detected_tissue_img_path = r"filtered_feature_bc_matrix\spatial\detected_tissue_image.jpg"
    # hires_img = plt.imread(hires_img_path)
    # detected_tissue_img = plt.imread(detected_tissue_img_path)

    spatial_dict = {}
    row_col_dict = {}
    points = []
    for row in spatial_info:
        spatial_dict[row[0]] = [row[-2], row[-1]]
        points.append([row[-2], row[-1]])
        row_col_dict[(row[2], row[3])] = row[0]

    colors = ["blue", "orange", "pink", "red", "yellow", "green", "cyan", "purple", "brown", "olive"]

    label_dict ={}

    for i in range(len(barcodes)):
    # Create a Rectangle patch
        label_dict[barcodes[i, 0]] = labels[i]

    neighbors_counts, neighbors_counts_by_class = spatial_analysis(row_col_dict, label_dict, n_clusters, colors)
    return barcodes, X_reduced, labels, colors[:n_clusters], spatial_dict, neighbors_counts, neighbors_counts_by_class, points

def vis_large(ax, ori_img_path, barcodes, class_labels, colors, spatial_dict, alpha):
    start= timeit.default_timer()
    ori_img = plt.imread(ori_img_path)
    # Display the image
    ax.imshow(ori_img)
    label_dict ={}
    for i in range(len(barcodes)):
    # Create a Rectangle patch
        label_dict[barcodes[i, 0]] = class_labels[i]
        center = spatial_dict[barcodes[i, 0]]
        color = colors[class_labels[i]]
        circle = patches.Circle((center[1], center[0]), 50, linewidth=1, edgecolor=color, facecolor=color, alpha=alpha)

        # Add the patch to the Axes
        ax.add_patch(circle)
        
    end = timeit.default_timer()
    print('vis ann img time:',str(end-start))


def vis_annotated_img(ax, small_img_path, barcodes, class_labels, colors, spatial_dict, alpha):
    scaling_factor=0.08250825
    #scaling_factor = 0.053097345
    #scaling_factor = 2000/24240
    full_d=177.4984743134119
    r=full_d*scaling_factor/2
    start= timeit.default_timer()
    small_img = plt.imread(small_img_path)

    # Display the image
    ax.imshow(small_img)
    label_dict ={}
    for i in range(len(barcodes)):
    # Create a Rectangle patch
        label_dict[barcodes[i, 0]] = class_labels[i]
        center = np.array(spatial_dict[barcodes[i, 0]])*scaling_factor
        center = center.astype('int')
        color = colors[class_labels[i]]
        circle = patches.Circle((center[1], center[0]), r, linewidth=1, edgecolor=color, facecolor=color, alpha=alpha)

        # Add the patch to the Axes
        ax.add_patch(circle)
        
    end = timeit.default_timer()
    print('vis small img time:',str(end-start))


def spatial_analysis(row_col_dict, label_dict, n_clusters, colors):
    
    start= timeit.default_timer()
    neighbors_counts = {}

    neighbors_counts_by_class = np.zeros((n_clusters, n_clusters))
    for k in range(n_clusters):
        neighbors_counts[k] = []


    for i in range(78):
        if i % 2 == 0:
            for j in range(0, 127, 2):
                neighbors = []
                if (i, j-2) in row_col_dict and row_col_dict[(i, j-2)] in label_dict:
                    neighbors.append(label_dict[row_col_dict[(i, j-2)]])
                if (i, j+2) in row_col_dict and row_col_dict[(i, j+2)] in label_dict:
                    neighbors.append(label_dict[row_col_dict[(i, j+2)]])
                if (i-1, j-1) in row_col_dict and row_col_dict[(i-1, j-1)] in label_dict:
                    neighbors.append(label_dict[row_col_dict[(i-1, j-1)]])
                if (i-1, j+1) in row_col_dict and row_col_dict[(i-1, j+1)] in label_dict:
                    neighbors.append(label_dict[row_col_dict[(i-1, j+1)]])
                if (i+1, j-1) in row_col_dict and row_col_dict[(i+1, j-1)] in label_dict:
                    neighbors.append(label_dict[row_col_dict[(i+1, j-1)]])
                if (i+1, j+1) in row_col_dict and row_col_dict[(i+1, j+1)] in label_dict:
                    neighbors.append(label_dict[row_col_dict[(i+1, j+1)]])

                counter = Counter(neighbors)
                if row_col_dict[(i, j)] in label_dict:
                    for k in counter:
                        neighbors_counts_by_class[label_dict[row_col_dict[(i, j)]]][k] += counter[k]
                count = list(counter.values())
                count /= np.sum(count)
                entropy_ = entropy(count, base=2)
                if row_col_dict[(i, j)] in label_dict:
                    neighbors_counts[label_dict[row_col_dict[(i, j)]]].append(entropy_)

        else:
            for j in range(1, 128, 2):
                neighbors = []
                if (i, j-2) in row_col_dict and row_col_dict[(i, j-2)] in label_dict:
                    neighbors.append(label_dict[row_col_dict[(i, j-2)]])
                if (i, j+2) in row_col_dict and row_col_dict[(i, j+2)] in label_dict:
                    neighbors.append(label_dict[row_col_dict[(i, j+2)]])
                if (i-1, j-1) in row_col_dict and row_col_dict[(i-1, j-1)] in label_dict:
                    neighbors.append(label_dict[row_col_dict[(i-1, j-1)]])
                if (i-1, j+1) in row_col_dict and row_col_dict[(i-1, j+1)] in label_dict:
                    neighbors.append(label_dict[row_col_dict[(i-1, j+1)]])
                if (i+1, j-1) in row_col_dict and row_col_dict[(i+1, j-1)] in label_dict:
                    neighbors.append(label_dict[row_col_dict[(i+1, j-1)]])
                if (i+1, j+1) in row_col_dict and row_col_dict[(i+1, j+1)] in label_dict:
                    neighbors.append(label_dict[row_col_dict[(i+1, j+1)]])

                counter = Counter(neighbors)
                if row_col_dict[(i, j)] in label_dict:
                    for k in counter:
                        neighbors_counts_by_class[label_dict[row_col_dict[(i, j)]]][k] += counter[k]
                count = list(counter.values())
                count /= np.sum(count)
                entropy_ = entropy(count, base=2)
                if row_col_dict[(i, j)] in label_dict:
                    neighbors_counts[label_dict[row_col_dict[(i, j)]]].append(entropy_)
    # max_ = np.max(list(neighbors_counts.values()))
                    
    end = timeit.default_timer()
    print('spatial analysis time:',str(end-start))
                    
    return neighbors_counts, neighbors_counts_by_class

def clustering(data, clustering_method, n_clusters):
    start= timeit.default_timer()
    if clustering_method == "KMeans":
        km = KMeans(n_clusters=n_clusters).fit(data)
        labels = km.labels_
    elif clustering_method == "GaussianMixture":
        gm = GaussianMixture(n_components=n_clusters).fit(data)
        labels = gm.predict(data)
    elif clustering_method == "HierarchicalClustering":
        dbscan = AgglomerativeClustering(n_clusters=n_clusters).fit(data)
        labels = dbscan.labels_
    elif clustering_method == "SpectralClustering":
        spectral = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize',).fit(data)
        labels = spectral.labels_
        
    end = timeit.default_timer()
    print('clustering time:',str(end-start))
    return labels

def normalization(data, nor):
    pass

def vis_neighbor_distribution(ax, n_clusters, neighbors_counts_by_class, colors):
    start= timeit.default_timer()
    labels = []
    for k in range(n_clusters):
        neighbors_counts_by_class[k] /= np.sum(neighbors_counts_by_class[k])
        labels.append("Class {}".format(k))

    ax.bar(labels, neighbors_counts_by_class[:, 0], 0.35, label="Class 0", color=colors[0])
    for k in range(1, n_clusters):
        ax.bar(labels, neighbors_counts_by_class[:, k], 0.35, bottom=np.sum(neighbors_counts_by_class[:, :k], axis=1),
        label="Class {}".format(k), color=colors[k])
    ax.set_ylabel("Fraction")
    ax.set_title("Neighbor Distribution")
    
    end = timeit.default_timer()
    print('vis neighbor dist time:',str(end-start))



def embedding_visualization(ax, embedding_method, X, y, color):
    start= timeit.default_timer()
    labels = []
    color_ = {}
    for k in y:
        labels.append("Class {}".format(k))
        color_["Class {}".format(k)] = color[k]

    if embedding_method == "TSNE":
        X_embedded = TSNE(n_components=2).fit_transform(X)
        embedding_title = "TSNE Embedding"
    elif embedding_method == "ISOMAP":
        X_embedded = Isomap(n_components=2).fit_transform(X)
        embedding_title = "ISOMAP Embedding"
    elif embedding_method == "SpectralEmbedding":
        X_embedded = SpectralEmbedding(n_components=2).fit_transform(X)
        embedding_title = "Spectral Embedding"
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=labels, palette=color_, ax = ax)
    ax.set_title(embedding_title)
    end = timeit.default_timer()
    print('vis embedding time:',str(end-start))


def violin_plot(ax, n_clusters, neighbors_counts, colors):
    start= timeit.default_timer()
    temp = []
    legend = []
    for k in range(n_clusters):
        legend.append("Class {}".format(k))
        temp.append(neighbors_counts[k])
    temp = np.array(temp, dtype=object)

    parts = ax.violinplot(temp, showmeans=False, showmedians=True)
    i = 0
    for pc in parts["bodies"]:
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.8)
        i += 1
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticks(np.arange(1, n_clusters + 1))
    ax.set_xticklabels(legend)
    ax.set_ylabel("Entropy")
    ax.set_title("Spatial Heterogeneity")
    
    end = timeit.default_timer()
    print('vis embedding time:',str(end-start))

def g_function(ax,class_labels,points,cluster):
    '''
    calculate the g_funcion of a cluster and plot in ax.
    higher means cluster
    g function measure whether the cluster also forms a spatial cluster
    inputs:
        ax: subgraph
        class_label: cluster labels
        points: 2d spatial coordinates of points
        cluster: int, the cluster to calculate g function for
    return:
        gp: (x,y,p_value,simulation)
    '''
    
    
    #get cluster points
    if (cluster==-1):
        cp=np.array(points)
    else:
        cp=[]
        for i in range(len(class_labels)):
            if (class_labels[i]==cluster):
                cp.append(points[i])
        cp=np.array(cp)
    
    
    gp1 = g_test(cp,keep_simulations=True,n_simulations=2)
    '''
    gp[0] is distance, the x-axis of the plot
    gp[1] is the g function of the corresponding distance, i.e. the y-axis
    gp[2] includes simulated values, corresponding to the y-axis of the 
    curve in the plot
    '''
    ax.plot(gp1[0],gp1[1])
    ax.plot(gp1[0],gp1[3][0])  
    ax.set_title("G function")
    #ax.plot(gp1[0],gp1[3][1])
    return gp1


def f_function(ax,class_labels,points,cluster):
    '''
    f function, cumulative density function of distance of the first nearest neighbor
    larger value at a distance means no cluster (dispersed)
    
    input:
        same as "g_function"
    return:
        same as "g_function"
    '''
    
    #get cluster points
    if (cluster==-1):
        cp=np.array(points)
    else:
        cp=[]
        for i in range(len(class_labels)):
            if (class_labels[i]==cluster):
                cp.append(points[i])
        cp=np.array(cp)
    
    f = f_test(cp,keep_simulations=True,n_simulations=2)
    ax.plot(f[0],f[1])
    ax.plot(f[0],f[3][0])  
    ax.set_title("F function")
    return f

def j_function(ax,class_labels,points,cluster):
    '''
    j function, "spatial hazard" function, combine G and F function
    larger value means dispersed
    
    input:
        same as "g_function"
    return:
        same as "g_function"
    '''
    
    #get cluster points
    if (cluster==-1):
        cp=np.array(points)
    else:
        cp=[]
        for i in range(len(class_labels)):
            if (class_labels[i]==cluster):
                cp.append(points[i])
        cp=np.array(cp)
    
    j = j_test(cp,keep_simulations=True,n_simulations=2)
    ax.plot(j[0],j[1])
    ax.plot(j[0],j[3][0])  
    ax.set_title("J function")
    return j

def k_function(ax,class_labels,points,cluster):
    '''
    k function, number of points closer than the distance
    smaller value at a distance means no cluster (dispersed)
    
    input:
        same as "g_function"
    return:
        same as "g_function"
    '''
    
    #get cluster points
    if (cluster==-1):
        cp=np.array(points)
    else:
        cp=[]
        for i in range(len(class_labels)):
            if (class_labels[i]==cluster):
                cp.append(points[i])
        cp=np.array(cp)
    
    k = k_test(cp,keep_simulations=True,n_simulations=2)
    ax.plot(k[0],k[1])
    ax.plot(k[0],k[3][0])  
    ax.set_title("K function")
    return k


def l_function(ax,class_labels,points,cluster):
    '''
    L function
    negative value suggests dispersion
    0 = random
    input:
        same as "g_function"
    return:
        same as "g_function"
    '''
    
    #get cluster points
    if (cluster==-1):
        cp=np.array(points)
    else:
        cp=[]
        for i in range(len(class_labels)):
            if (class_labels[i]==cluster):
                cp.append(points[i])
        cp=np.array(cp)
    
    l = l_test(cp,keep_simulations=True,n_simulations=2)
    ax.plot(l[0],l[1])
    ax.plot(l[0],l[3][0])  
    ax.set_title("L function")
    return l

def min_nnd(class_labels,points,cluster):
    '''
    mean nearest neighbor statistics
    input:
        class_labels: cluster labels
        points: spatial positions of cells
        cluster: the cluster to calculate the statistics for, -1 = all
    return:
        min_nnd statistics
    '''
    
    #get cluster points
    if (cluster==-1):
        cp=np.array(points)
    else:
        cp=[]
        for i in range(len(class_labels)):
            if (class_labels[i]==cluster):
                cp.append(points[i])
        cp=np.array(cp)
        

    pp=PointPattern(cp)
    
    print("mean nn distance: ",pp.mean_nnd)
    
    return pp.mean_nnd 
    

if __name__ == "__main__":
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ori_img_path = r"filtered_feature_bc_matrix\spatial\detected_tissue_image.jpg"
    matrix_path = r"filtered_feature_bc_matrix\matrix.mtx.gz"
    features_path = r"filtered_feature_bc_matrix\features.tsv.gz"
    barcodes_path = r"filtered_feature_bc_matrix\barcodes.tsv.gz"
    spatial_list_path = r"filtered_feature_bc_matrix\spatial\tissue_positions_list.csv"
    large_img_path =  r"filtered_feature_bc_matrix\Targeted_Visium_Human_BreastCancer_Immunology_image.tif"
    n_clusters = 6
    alpha = 0.6

    clustering_method = "HierarchicalClustering"

    embedding_method = "SpectralEmbedding"
    
    # main
    barcodes, X_reduced, class_labels, colors, spatial_dict, neighbors_counts, neighbors_counts_by_class,points = main(clustering_method, n_clusters,
    ori_img_path, matrix_path, features_path, barcodes_path, spatial_list_path)



    # statistics
    #min_nnd(class_labels,points,cluster)


    # g function
    gplot=plt.subplot()
    cluster=3
    #g = g_function(gplot,class_labels,points,cluster)
    #f = f_function(gplot,class_labels,points,cluster)
    #k = k_function(gplot,class_labels,points,cluster)
    #j = j_function(gplot,class_labels,points,cluster)
    l = l_function(gplot,class_labels,points,cluster)
    #vis_annotated_img(gplot, ori_img_path, barcodes, class_labels, colors, spatial_dict, alpha)
    plt.show()
      
    
    
    
    
      
    # visualization
    '''
    vis_annotated_img(ax1, ori_img_path, barcodes, class_labels, colors, spatial_dict, alpha)
    embedding_visualization(ax2, embedding_method, X_reduced, class_labels, colors)
    vis_neighbor_distribution(ax3, n_clusters, neighbors_counts_by_class, colors)
    violin_plot(ax4, n_clusters, neighbors_counts, colors)
    plt.show()
    '''

    