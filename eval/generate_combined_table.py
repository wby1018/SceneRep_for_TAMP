import pandas as pd
import numpy as np

def process_csv(path):
    df = pd.read_csv(path)
    df['add_success_rate'] *= 100
    df['adds_success_rate'] *= 100
    df_grouped = df.groupby(['method', 'object_name'])[['add_success_rate', 'adds_success_rate']].mean().reset_index()
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
    return pivot, method_order

pivot_single, methods_s = process_csv('/home/wby/baselines/SR_TAMP/eval/global_results_single.csv')
pivot_multi, methods_m = process_csv('/home/wby/baselines/SR_TAMP/eval/global_results_multi.csv')

method_order = methods_s

print('\\begin{table*}[t]')
print('\\centering')
print('\\caption{Detailed Tracking Performance on Single and Multi-Object Scenarios. Each cell reports ADD / ADD-S in percentage.}')
print('\\label{tab:detailed_results}')
print('% \\resizebox{\\textwidth}{!}{')
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
print('\\multicolumn{' + str(len(method_order)+1) + '}{c}{\\textbf{Single-Object Scenarios}} \\\\')
print('\\midrule')

for idx, row in pivot_single.iterrows():
    row_name = str(idx).replace('_', ' ').title() if idx != 'Avg.' else '\\midrule\n\\textbf{Avg.}'
    cells = list(row.values)
    print(f"{row_name} & " + " & ".join(cells) + " \\\\")

print('\\midrule')
print('\\multicolumn{' + str(len(method_order)+1) + '}{c}{\\textbf{Multi-Object Scenarios}} \\\\')
print('\\midrule')

for idx, row in pivot_multi.iterrows():
    row_name = str(idx).replace('_', ' ').title()
    # If the object name is e.g. "Apple1", title() makes it "Apple1"
    # Actually let's manually fix "Apple1"-> "Apple 1"
    import re
    row_name = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', row_name)
    if idx == 'Avg.':
        row_name = '\\midrule\n\\textbf{Avg.}'
    cells = list(row.values)
    print(f"{row_name} & " + " & ".join(cells) + " \\\\")

print('\\bottomrule')
print('\\end{tabular}')
print('% }')
print('\\end{table*}')
