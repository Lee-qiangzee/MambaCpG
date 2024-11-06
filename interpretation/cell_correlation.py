import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap

folder1 = 'methylation_files'
folder2 = 'unmethylation_files'
files1 = os.listdir(folder1)
files2 = os.listdir(folder2)

files1_dict = {fname[1:]: fname for fname in files1}
files2_dict = {fname[1:]: fname for fname in files2}

common_bases = set(files1_dict.keys()).intersection(set(files2_dict.keys()))
common_bases = sorted(common_bases, key=lambda x: int(x.split('.')[0]))

all_methylation = []
all_unmethylation = []
for base_name in common_bases:
    file1 = os.path.join(folder1, files1_dict[base_name])
    file2 = os.path.join(folder2, files2_dict[base_name])
    data1 = np.load(file1)['total']
    data2 = np.load(file2)['total']
    
    positive_sum1 = np.sum(np.where(data1 > 0, data1, 0), axis=(0, 1))
    negative_sum1 = np.sum(np.where(data1 < 0, np.abs(data1), 0), axis=(0, 1))
    positive_sum2 = np.sum(np.where(data2 > 0, data2, 0), axis=(0, 1))
    negative_sum2 = np.sum(np.where(data2 < 0, np.abs(data2), 0), axis=(0, 1))
    
    unmethylation = (positive_sum1 + negative_sum2) + 1e-4
    methylation = (positive_sum2 + negative_sum1) + 1e-4

    all_methylation.append(methylation)
    all_unmethylation.append(unmethylation)

all_methylation_array = np.array(all_methylation)  
all_unmethylation_array = np.array(all_unmethylation)

def min_max_normalize(data):
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max > data_min:
        return (data - data_min) / (data_max - data_min)
    else:
        return np.zeros_like(data)

all_methylation_array = min_max_normalize(all_methylation_array) + 1e-4
all_unmethylation_array = min_max_normalize(all_unmethylation_array) + 1e-4

colors = [
    (0.0, 'royalblue'),
    (0.2, 'dodgerblue'),
    (0.6, 'mediumspringgreen'),
    (1.0, 'yellow')
]
custom_cmap = LinearSegmentedColormap.from_list('custom_blue_green_yellow', colors)

def plot_heatmap(data, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap=custom_cmap, interpolation='nearest', origin='lower', norm=LogNorm(vmin=0.01, vmax=0.2))
    plt.colorbar(label='Value')
    plt.xlabel('Cell Index')
    plt.ylabel('Cell Index')
    plt.xticks(np.arange(0, 30, 1), rotation=90)
    plt.yticks(np.arange(0, 30, 1))

    regions = [(0, 3), (4, 6), (7, 10), (11, 13), (14, 17), (18, 21), (22, 25), (26, 29)]

    for (start, end) in regions:
        plt.gca().add_patch(plt.Rectangle((start - 0.5, start - 0.5), end - start + 1, end - start + 1,
                                          fill=False, edgecolor='black', lw=2))

    plt.savefig(filename)
    plt.close()

plot_heatmap(all_methylation_array, 'all_methylation_array_heatmap.pdf')
plot_heatmap(all_unmethylation_array, 'all_unmethylation_array_heatmap.pdf')