import pandas as pd

def process_csv(path):
    df = pd.read_csv(path)
    df['add_success_rate'] *= 100
    df['adds_success_rate'] *= 100
    
    metrics = ['add_success_rate', 'adds_success_rate', 't_err_mean', 'r_err_mean']
    
    df_grouped = df.groupby(['method', 'object_name'])[metrics].mean().reset_index()
    df_avg = df.groupby(['method'])[metrics].mean().reset_index()
    df_avg['object_name'] = 'Avg.'
    
    df_all = pd.concat([df_grouped, df_avg], ignore_index=True)
    
    # Format: ADD / ADD-S / E_t / E_r
    df_all['cell'] = df_all.apply(
        lambda row: f"{row['add_success_rate']:.1f} / {row['adds_success_rate']:.1f} / {row['t_err_mean']:.2f} / {row['r_err_mean']:.1f}", axis=1
    )
    
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

print('\\begin{table*}[t]')
print('\\centering')
print('\\caption{Detailed Tracking Performance on Single and Multi-Object Scenarios. Each cell reports \\textbf{ADD (\\%) / ADD-S (\\%) / $E_t$ (cm) / $E_R$ ($^\\circ$)}.}')
print('\\label{tab:detailed_results}')
print('\\resizebox{\\textwidth}{!}{')
print('\\begin{tabular}{l' + 'c' * len(methods_s) + '}')
print('\\toprule')

headers = []
for m in methods_s:
    headers.append('\\textbf{\\ours}' if m == 'Ours' else m)

print('\\textbf{Object} & ' + ' & '.join(headers) + ' \\\\')
print('\\midrule')
print('\\multicolumn{' + str(len(methods_s)+1) + '}{c}{\\textbf{Single-Object Scenarios}} \\\\')
print('\\midrule')

for idx, row in pivot_single.iterrows():
    row_name = str(idx).replace('_', ' ').title() if idx != 'Avg.' else '\\midrule\n\\textbf{Avg.}'
    print(f"{row_name} & " + " & ".join(row.values) + " \\\\")

print('\\midrule')
print('\\multicolumn{' + str(len(methods_s)+1) + '}{c}{\\textbf{Multi-Object Scenarios}} \\\\')
print('\\midrule')

import re
for idx, row in pivot_multi.iterrows():
    row_name = str(idx).replace('_', ' ').title()
    row_name = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', row_name)
    if idx == 'Avg.':
        row_name = '\\midrule\n\\textbf{Avg.}'
    print(f"{row_name} & " + " & ".join(row.values) + " \\\\")

print('\\bottomrule')
print('\\end{tabular}')
print('}')
print('\\end{table*}')
