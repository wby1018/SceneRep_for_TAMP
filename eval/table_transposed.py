import pandas as pd

df = pd.read_csv('/home/wby/baselines/SR_TAMP/eval/global_results_single.csv')
df['add_success_rate'] *= 100
df['adds_success_rate'] *= 100

# group by method and object
df_grouped = df.groupby(['method', 'object_name'])[['add_success_rate', 'adds_success_rate']].mean().reset_index()

# group all objects by method to get averages
df_avg = df.groupby(['method'])[['add_success_rate', 'adds_success_rate']].mean().reset_index()
df_avg['object_name'] = 'Avg.'

df_all = pd.concat([df_grouped, df_avg], ignore_index=True)
df_all['cell'] = df_all.apply(lambda row: f"{row['add_success_rate']:.1f} / {row['adds_success_rate']:.1f}", axis=1)

pivot = df_all.pivot(index='object_name', columns='method', values='cell')

method_order = ['TSDF++', 'MidFusion', 'BundleSDF', 'FoundationPose', 'Ours']
method_order = [m for m in method_order if m in pivot.columns]
pivot = pivot[method_order]

objects = sorted([c for c in pivot.index if c != 'Avg.'])
row_order = objects + ['Avg.']
pivot = pivot.reindex(row_order)

print('\\begin{table}[t]')
print('\\centering')
print('\\caption{Detailed Single Object Tracking Performance. Each cell reports ADD / ADD-S in percentage.}')
print('\\label{tab:single_obj_detailed}')
print('\\resizebox{\\linewidth}{!}{')
print('\\begin{tabular}{l' + 'c' * len(method_order) + '}')
print('\\toprule')

headers = []
for m in method_order:
    if m == 'Ours':
        headers.append('\\textbf{\\ours}')
    else:
        headers.append(m)

print('\\textbf{Object} & ' + ' & '.join(headers) + ' \\\\')
print('\\midrule')

for idx, row in pivot.iterrows():
    row_name = idx.replace('_', ' ').title() if idx != 'Avg.' else '\\midrule\n\\textbf{Avg.}'
    cells = list(row.values)
    print(f"{row_name} & " + " & ".join(cells) + " \\\\")

print('\\bottomrule')
print('\\end{tabular}')
print('}')
print('\\end{table}')
