import pandas as pd
import difflib
import sys
import os

def main():
    threshold_file = 'success_threshold.txt'
    if not os.path.exists(threshold_file):
        print(f"Error: {threshold_file} not found.")
        return

    thresholds = {}
    with open(threshold_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line == '.':
                continue
            parts = line.split(':')
            if len(parts) == 2:
                name = parts[0].strip()
                val = int(parts[1].strip())
                thresholds[name] = val
                
    valid_names = list(thresholds.keys())
    
    csv_files = ['global_results_single.csv', 'global_results_multi.csv']
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"File {csv_file} not found, skipping.")
            continue
            
        print(f"Processing {csv_file}...")
        df = pd.read_csv(csv_file)
        
        updated_rows = 0
        for idx in range(len(df)):
            obj_name = str(df.at[idx, 'object_name'])
            
            # First try exact subset match (e.g., 'apple' in 'apple_1')
            match_name = None
            for v in valid_names:
                if v in obj_name:
                    match_name = v
                    break
            
            # Fallback to difflib
            if not match_name:
                matches = difflib.get_close_matches(obj_name, valid_names, n=1, cutoff=0.1)
                if matches:
                    match_name = matches[0]
                
            if not match_name:
                print(f"  Warning: Could not find matching threshold for {obj_name}")
                continue
                
            thresh_mm = thresholds[match_name]
            add_col = f"add_sr_{thresh_mm}mm"
            adds_col = f"adds_sr_{thresh_mm}mm"
            
            if add_col in df.columns and adds_col in df.columns:
                old_add = df.at[idx, 'add_success_rate']
                old_adds = df.at[idx, 'adds_success_rate']
                new_add = df.at[idx, add_col]
                new_adds = df.at[idx, adds_col]
                df.at[idx, 'add_success_rate'] = new_add
                df.at[idx, 'adds_success_rate'] = new_adds
                print(f"Row {idx} [{obj_name}]: threshold={thresh_mm}mm, ADD {old_add} -> {new_add}, ADD-S {old_adds} -> {new_adds}")
                updated_rows += 1
            else:
                print(f"  Warning: Columns {add_col} or {adds_col} not found in {csv_file}")
                
        # Save back
        df.to_csv(csv_file, index=False)
        print(f"Saved {csv_file}. Updated {updated_rows} rows.")

if __name__ == "__main__":
    main()
