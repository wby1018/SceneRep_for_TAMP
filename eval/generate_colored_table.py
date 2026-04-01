import pandas as pd
import numpy as np
import re

def process_data(path, is_multi=False):
    df = pd.read_csv(path)
    df['add_success_rate'] *= 100
    df['adds_success_rate'] *= 100
    
    metrics = ['add_success_rate', 'adds_success_rate', 't_err_mean', 'r_err_mean']
    
    df_grouped = df.groupby(['method', 'object_name'])[metrics].mean().reset_index()
    df_avg = df.groupby(['method'])[metrics].mean().reset_index()
    df_avg['object_name'] = 'Avg.'
    
    df_all = pd.concat([df_grouped, df_avg], ignore_index=True)
    
    if is_multi:
        def fix_name(name):
            if name == 'Avg.': return name
            name = name.replace('_', ' ').title()
            name = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', name)
            return name
        df_all['object_name'] = df_all['object_name'].apply(fix_name)
    else:
        df_all['object_name'] = df_all['object_name'].apply(lambda x: x.replace('_', ' ').title() if x != 'Avg.' else x)
        
    return df_all

df_s = process_data('/home/wby/baselines/SR_TAMP/eval/global_results_single.csv', False)
df_m = process_data('/home/wby/baselines/SR_TAMP/eval/global_results_multi.csv', True)

methods = ['TSDF++', 'MidFusion', 'BundleSDF', 'FoundationPose', 'Ours']

def get_table_rows(df, objects):
    rows_latex = []
    for obj in objects:
        df_obj = df[df['object_name'] == obj]
        
        add_vals = {}
        adds_vals = {}
        t_err_vals = {}
        r_err_vals = {}
        
        for m in methods:
            row = df_obj[df_obj['method'] == m]
            if len(row) > 0:
                add_vals[m] = round(row['add_success_rate'].values[0], 1)
                adds_vals[m] = round(row['adds_success_rate'].values[0], 1)
                t_err_vals[m] = round(row['t_err_mean'].values[0], 2)
                r_err_vals[m] = round(row['r_err_mean'].values[0], 1)
            else:
                add_vals[m] = None
                adds_vals[m] = None
                t_err_vals[m] = None
                r_err_vals[m] = None
                
        valid_add = sorted(list(set([v for v in add_vals.values() if v is not None])), reverse=True)
        valid_adds = sorted(list(set([v for v in adds_vals.values() if v is not None])), reverse=True)
        valid_t = sorted(list(set([v for v in t_err_vals.values() if v is not None])))
        valid_r = sorted(list(set([v for v in r_err_vals.values() if v is not None])))
        
        best_add = valid_add[0] if len(valid_add) > 0 else -1
        sec_add = valid_add[1] if len(valid_add) > 1 else -1

        best_adds = valid_adds[0] if len(valid_adds) > 0 else -1
        sec_adds = valid_adds[1] if len(valid_adds) > 1 else -1
        
        best_t = valid_t[0] if len(valid_t) > 0 else -1
        sec_t = valid_t[1] if len(valid_t) > 1 else -1

        best_r = valid_r[0] if len(valid_r) > 0 else -1
        sec_r = valid_r[1] if len(valid_r) > 1 else -1
        
        row_str = f"{obj}"
        for m in methods:
            v_add = add_vals[m]
            v_adds = adds_vals[m]
            v_t = t_err_vals[m]
            v_r = r_err_vals[m]
            
            if v_add is not None:
                bg_add = "\\cellcolor{blue!25}" if v_add == best_add else "\\cellcolor{blue!10}" if v_add == sec_add else ""
                bg_adds = "\\cellcolor{blue!25}" if v_adds == best_adds else "\\cellcolor{blue!10}" if v_adds == sec_adds else ""
                bg_t = "\\cellcolor{blue!25}" if v_t == best_t else "\\cellcolor{blue!10}" if v_t == sec_t else ""
                bg_r = "\\cellcolor{blue!25}" if v_r == best_r else "\\cellcolor{blue!10}" if v_r == sec_r else ""
                
                s_add = f"{bg_add}{v_add:.1f}"
                s_adds = f"{bg_adds}{v_adds:.1f}"
                s_t = f"{bg_t}{v_t:.2f}"
                s_r = f"{bg_r}{v_r:.1f}"
                row_str += f" & {s_add} & {s_adds} & {s_t} & {s_r}"
            else:
                row_str += " & - & - & - & -"
        row_str += " \\\\"
        rows_latex.append(row_str)
    return rows_latex

objs_s = sorted([x for x in df_s['object_name'].unique() if x != 'Avg.'])
objs_m = sorted([x for x in df_m['object_name'].unique() if x != 'Avg.'])

rows_s = get_table_rows(df_s, objs_s)
avg_s = get_table_rows(df_s, ['Avg.'])
rows_m = get_table_rows(df_m, objs_m)
avg_m = get_table_rows(df_m, ['Avg.'])

print('\\begin{table*}[t]')
print('\\centering')
print('\\caption{Detailed Tracking Performance. The best results are highlighted in \\colorbox{blue!25}{dark blue}, and the second best in \\colorbox{blue!10}{light blue}.}')
print('\\label{tab:detailed_results}')
print('\\resizebox{\\textwidth}{!}{')
print('\\begin{tabular}{l | cccc | cccc | cccc | cccc | cccc}')
print('\\toprule')

headers1 = ['\\textbf{Object}']
headers2 = ['']
for m in methods:
    if m == 'Ours':
        headers1.append('\\multicolumn{4}{c}{\\textbf{\\ours}}')
    else:
        headers1.append(f'\\multicolumn{{4}}{{c}}{{{m}}}')
    headers2.extend(['ADD$\\uparrow$', 'ADD-S$\\uparrow$', '$E_t$$\\downarrow$', '$E_R$$\\downarrow$'])

print(' & '.join(headers1) + ' \\\\')
print(' & '.join(headers2) + ' \\\\')
print('\\midrule')
print('\\multicolumn{21}{c}{\\textbf{Single-Object Scenarios}} \\\\')
print('\\midrule')

for r in rows_s:
    print(r)
print('\\midrule')
for r in avg_s:
    print(r.replace('Avg.', '\\textbf{Avg.}'))

print('\\midrule')
print('\\multicolumn{21}{c}{\\textbf{Multi-Object Scenarios}} \\\\')
print('\\midrule')

for r in rows_m:
    print(r)
print('\\midrule')
for r in avg_m:
    print(r.replace('Avg.', '\\textbf{Avg.}'))

print('\\bottomrule')
print('\\end{tabular}')
print('}')
print('\\end{table*}')
