import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec

file_path = '.npz' # The output file of integrated_gradients.py
data = np.load(file_path)

total_for_plot = np.squeeze(data['total'], axis=0)

n_cpg_sites = total_for_plot.shape[0]
halfway_index = n_cpg_sites // 2
x_labels = np.arange(-halfway_index, halfway_index + 1)

fig = plt.figure(figsize=(50, 10))

gs = GridSpec(2, 2, width_ratios=[0.95, 0.05], height_ratios=[0.2, 0.8], wspace=0.005, hspace=0.01)

ax_joint = fig.add_subplot(gs[1, 0])
sns.heatmap(total_for_plot.T, cmap="RdBu", ax=ax_joint, cbar=False, center=0, vmin=-0.02, vmax=0.02)

ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
row_sums = np.sum(np.abs(total_for_plot), axis=1)
ax_marg_x.bar(np.arange(n_cpg_sites) + 0.5, row_sums, color='blue', align='center')
ax_marg_x.axis('off')

ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)
col_sums = np.sum(np.abs(total_for_plot), axis=0)
ax_marg_y.barh(np.arange(total_for_plot.shape[1]) + 0.5, col_sums, color='blue', align='center')
ax_marg_y.axis('off')

cax = ax_joint.inset_axes([0.05, 0.1, 0.02, 0.8])
cbar = plt.colorbar(ax_joint.collections[0], cax=cax, orientation='vertical')
cbar.set_label("Contribution")

ax_joint.set_xticks(np.arange(0, n_cpg_sites, step=5))
ax_joint.set_xticklabels(x_labels[::5])

ax_joint.set_xlabel("CpG Sites")
ax_joint.set_ylabel("Cells")

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)

output_path = 'contribution_matrix.pdf'
plt.savefig(output_path, format='pdf', dpi=600, bbox_inches='tight')
plt.close()

data.close()

print("Done")
