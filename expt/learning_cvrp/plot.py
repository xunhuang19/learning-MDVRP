import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and process data
data_dir = '../../data/results50'
save_dir = 'plots/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

hga_files = [file_name for file_name in os.listdir(data_dir) if "hga.csv" in file_name]
nni_files = [file_name for file_name in os.listdir(data_dir) if "nni.csv" in file_name]
lga_files = [file_name for file_name in os.listdir(data_dir) if "lga.csv" in file_name]
non_collab_file = [file_name for file_name in os.listdir(data_dir) if "benefit" in file_name]

hga_files = [os.path.join(data_dir, file_name) for file_name in hga_files]
lga_files = [os.path.join(data_dir, file_name) for file_name in lga_files]
nni_files = [os.path.join(data_dir, file_name) for file_name in nni_files]
non_collab_file = [os.path.join(data_dir, file_name) for file_name in non_collab_file]

hga_df = pd.concat([pd.read_csv(file, index_col=[0, 1]) for file in hga_files])
nni_df = pd.concat([pd.read_csv(file, index_col=[0, 1]) for file in nni_files])
lga_df = pd.concat([pd.read_csv(file, index_col=[0, 1]) for file in lga_files])
non_collab = pd.concat([pd.read_csv(file, index_col=0) for file in non_collab_file])

hga_gp = hga_df.groupby(level=["Instance"])
hga_stat = hga_gp.describe()

assert lga_df.index.equals(nni_df.index)

nni_gp = nni_df.groupby(level=["Instance"])
nni_stat = nni_gp.describe()
nni_stat = nni_stat.sort_values(by=nni_stat.columns[9])

lga_gp = lga_df.groupby(level=["Instance"])
lga_stat = lga_gp.describe()
lga_stat = lga_stat.reindex(nni_stat.index)
hga_stat = hga_stat.reindex(nni_stat.index)

lga_df.insert(3, "Instance No.", "")
lga_df.insert(4, "n_customer", "")
lga_df.insert(5, "n_depot", "")
nni_df.insert(3, "Instance No.", "")

for i, index in enumerate(lga_df.index):
    lga_df.iloc[i, lga_df.columns.get_loc('Instance No.')] = lga_stat.index.get_loc(index[0])
    if '20' in index[0]:
        lga_df.iloc[i, lga_df.columns.get_loc('n_customer')] = 20
    elif '50' in index[0]:
        lga_df.iloc[i, lga_df.columns.get_loc('n_customer')] = 50
    else:
        lga_df.iloc[i, lga_df.columns.get_loc('n_customer')] = 100
    if '2D' in index[0]:
        lga_df.iloc[i, lga_df.columns.get_loc('n_depot')] = 2
    else:
        lga_df.iloc[i, lga_df.columns.get_loc('n_depot')] = 4

lga_df_reset = lga_df.reset_index().set_index('Instance')
lga_df_reset = lga_df_reset.join(non_collab).set_index('shuffle time', append=True)
lga_df_reset['Saving'] = (lga_df_reset['Non-collaborative cost'] - lga_df_reset['cost']) / lga_df_reset[
    'Non-collaborative cost'] * 100
lga_100c, lga_50c, lga_20c = lga_df_reset[lga_df_reset['n_customer'] == 100], lga_df_reset[
    lga_df_reset['n_customer'] == 50], lga_df_reset[
    lga_df_reset['n_customer'] == 20]
lga_2d, lga_4d = lga_df_reset[lga_df_reset['n_depot'] == 2], lga_df_reset[lga_df_reset['n_depot'] == 4]

nni_df['Instance No.'] = lga_df['Instance No.']

line_X = lga_df["Instance No."]
line_X_mean = np.array(range(len(lga_stat)))

fig1 = plt.figure()
ax = fig1.add_subplot()

ax.scatter(line_X, lga_df['performance'], s=12, alpha=0.3, edgecolors="k", c="#04D8B2", label="LGA-NNI")
ax.scatter(line_X, nni_df['performance'], s=12, alpha=0.3, edgecolors="k", c="#DBB40C", label="NNI")

ax.plot(line_X_mean, lga_stat["performance", "mean"], color="#04D8B2", label="LGA-NNI Mean")
ax.plot(line_X_mean, nni_stat["performance", "mean"], color="#DBB40C", label="NNI Mean")
ax.plot(line_X_mean, [1] * len(line_X_mean), 'r--', linewidth=2, label="HGA Mean")

ax.fill_between(line_X_mean, lga_stat["performance", "25%"], lga_stat["performance", "75%"],
                alpha=0.3, color="#7FFFD4", label="LGA-NNI Interquartile Range")
ax.set_xlabel("Instance No.")
ax.set_ylabel("Performance Index")
plt.xticks(np.arange(len(line_X_mean)), np.arange(1, len(line_X_mean) + 1))
ax.legend()
ax.yaxis.grid(True)

legend = ax.get_legend()
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((0, 0, 1, 0.1))
fig1.tight_layout()
fig1.show()
fig1.savefig(f"{save_dir}/Algorithm_comparison.png", dpi=600, transparent=True)

data = [lga_20c['performance'], lga_50c['performance'], lga_100c['performance']]
fig2 = plt.figure()
ax = fig2.add_subplot()
bplot = ax.boxplot(data, notch=True,
                   vert=True,
                   patch_artist=True, capprops=dict(linewidth=2),
                   medianprops=dict(color="darkorange", linewidth=1.5))  # fill with color

colors = ['pink', 'lightblue', 'lightgreen']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xticklabels(['20', '50', '100'])
scatter_colour = ['red', 'dodgerblue', 'seagreen']
for i in [1, 2, 3]:
    y = data[i - 1]
    x = np.random.normal(i, 0.04, size=len(y))
    ax.plot(x, y, '.', color=scatter_colour[i - 1], alpha=0.2)

ax.set_ylabel('Performance Index')
ax.set_xlabel('Number of Customers')
ax.yaxis.grid(True)
fig2.tight_layout()
fig2.show()

fig2.savefig(f"{save_dir}/network_size_performance.png", dpi=600, transparent=True)

benefit_stat = lga_stat.copy()
benefit_stat[non_collab.columns] = non_collab

fig3 = plt.figure()
ax = fig3.add_subplot()
colors = ["#849b91", "#99857e", "#b4746b", "#C2CEDC"]  #
ax.plot(line_X_mean, -benefit_stat["cost", "mean"], label="After Collaboration", color='r')
ax.plot(line_X_mean, -benefit_stat["Non-collaborative cost"], label="No Collaboration",
        color='k')
plt.xticks(np.arange(len(line_X_mean)), np.arange(1, len(line_X_mean) + 1), fontsize=13)
ax.set_ylabel("Utility [-Travel Cost]", fontsize=13)
ax.set_xlabel("Instance No.", fontsize=13)
ax.legend(fontsize=13)

legend = ax.get_legend()
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((0, 0, 1, 0.1))
ax.yaxis.grid(True)
fig3.set_figwidth(8)
fig3.show()
fig3.tight_layout()
fig3.savefig(f"{save_dir}/benefit_assessment.png", dpi=600, transparent=True)

fig4 = plt.figure()
ax = fig4.add_subplot()
data = [lga_2d['Saving'], lga_4d['Saving']]
bplot = ax.boxplot(data, vert=True, notch=True, capprops=dict(linewidth=2),
                   medianprops=dict(color="darkorange", linewidth=1.5))  # vertical box alignment
for median in bplot['medians']:
    median.set(color='red')

ax.set_xticklabels([2, 4])
ax.set_ylabel('Saving [%]')
ax.set_xlabel('Number of Depots')
ax.yaxis.grid(True)

scatter_colour = ['green', 'deepskyblue']
transp = [0.2, 0.1]
for i in [1, 2]:
    y = data[i - 1]
    x = np.random.normal(i, 0.02, size=len(y))
    ax.plot(x, y, '.', color=scatter_colour[i - 1], alpha=transp[i - 1])

fig4.tight_layout()
fig4.show()
fig4.savefig(f"{save_dir}/depot_saving.png", dpi=600, transparent=True)

data = [lga_20c['Saving'], lga_50c['Saving'], lga_100c['Saving']]
fig5 = plt.figure()
ax = fig5.add_subplot()
bplot = ax.boxplot(data, notch=True,
                   vert=True,
                   patch_artist=True, capprops=dict(linewidth=2),
                   medianprops=dict(color="darkorange", linewidth=1.5))  # fill with color

colors = ['pink', 'lightblue', 'lightgreen']

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
scatter_colour = ['red', 'dodgerblue', 'seagreen']
for i in [1, 2, 3]:
    y = data[i - 1]
    x = np.random.normal(i, 0.04, size=len(y))
    ax.plot(x, y, '.', color=scatter_colour[i - 1], alpha=0.2)

ax.set_xticklabels(['20', '50', '100'])
ax.set_ylabel('Saving [%]')
ax.set_xlabel('Number of Customers')
ax.yaxis.grid(True)

fig5.tight_layout()
fig5.show()
fig5.savefig(f"{save_dir}/network_size_saving.png", dpi=600, transparent=True)
