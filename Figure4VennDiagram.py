from venn import venn
import matplotlib.pyplot as plt

Stacking = {
    "MAP2K3",
    "SLC36A1",
    "RCN2",
    "SMN1",
    "RPS7",
    "RPL29",
    "IL1R2",
    "ENSG00000285272",
    "ARL4C",
    "CHCHD10",
    "UBE2Q2",
    "KLHL21",
    "LSM12",
    "STEAP4",
    "EEF1AKMT2",
    "GPR180",
    "ACTN1",
    "DCBLD1",
    "SUSD1",
    "CERS5"
}

Voting = {
    "SLC36A1",
    "MAP2K3",
    "RPL29",
    "RCN2",
    "RPS7",
    "SMN1",
    "ENSG00000282572",
    "IL1R2",
    "ACTN1",
    "ARL4C",
    "LSM12",
    "STEAP4",
    "KLHL21",
    "CHCHD10",
    "UBE2O",
    "GPR180",
    "EEF1AKMT2",
    "MTMR3",
    "RBMX",
    "DCBLD1"}

XGBoost = {"SLC36A1",
    "MAP2K3",
    "RPL29",
    "ENSG00000282572",
    "SMN1",
    "RCN2",
    "IL1R2",
    "ACTN1",
    "EEF1AKMT2",
    "CHD10",
    "ARL4C",
    "POPS",
    "PRPF18",
    "ENSG00000276533",
    "UTP18",
    "IL17RA",
    "SETA4",
    "RBMX",
    "DENND3",
    "GPR180"
}



RandomForest = {
     "SLC36A1",
    "ENSG00000282572",
    "MAP2K3",
    "RPL29",
    "RBMX",
    "ACTN1",
    "MTMR3",
    "RCN2",
    "RPL21",
    "ZFYVE1",
    "EEF1AKMT2",
    "RPS7",
    "DENND3",
    "MAVS",
    "IL17RA",
    "CHCHD10",
    "SMN1",
    "UTP18",
    "TLE5",
    "P0P5"
}


dictionary = {
    "Stacking Ensemble": Stacking,
    "Voting Ensemble": Voting,
    "XGBoost": XGBoost,
    "Random Forest": RandomForest}

intersections = {
    '1000': Stacking - Voting - XGBoost - RandomForest,
    '0100': Voting - Stacking - XGBoost - RandomForest,
    '0010': XGBoost - Stacking - Voting - RandomForest,
    '0001': RandomForest - Stacking - Voting - XGBoost,
    '1100': (Stacking & Voting) - XGBoost - RandomForest,
    '1010': (Stacking & XGBoost) - Voting - RandomForest,
    '1001': (Stacking & RandomForest) - Voting - XGBoost,
    '0110': (Voting & XGBoost) - Stacking - RandomForest,
    '0101': (Voting & RandomForest) - Stacking - XGBoost,
    '0011': (XGBoost & RandomForest) - Stacking - Voting,
    '1110': (Stacking & Voting & XGBoost) - RandomForest,
    '1101': (Stacking & Voting & RandomForest) - XGBoost,
    '1011': (Stacking & XGBoost & RandomForest) - Voting,
    '0111': (Voting & XGBoost & RandomForest) - Stacking,
    '1111': Stacking & Voting & XGBoost & RandomForest
}

### Export intersections to files and summary CSV
import os
import pandas as pd

out_dir = os.path.join('graph', 'venn_regions')
os.makedirs(out_dir, exist_ok=True)

# human-readable names corresponding to bit order: Stacking, Voting, XGBoost, RandomForest
bit_names = ['Stacking_Ensemble', 'Voting_Ensemble', 'XGBoost', 'Random_Forest']

rows = []
for code, genes in intersections.items():
    genes_sorted = sorted(genes)
    # build readable region name from bits
    bits = list(code)
    members = [bit_names[i] for i, b in enumerate(bits) if b == '1']
    region_name = '__AND__'.join(members) if members else 'None'

    # write genes to text file
    safe_fname = f"region_{code}_{region_name}.txt".replace(' ', '_')
    path = os.path.join(out_dir, safe_fname)
    with open(path, 'w') as fh:
        for g in genes_sorted:
            fh.write(g + '\n')

    rows.append({'RegionCode': code, 'RegionName': region_name, 'Count': len(genes_sorted), 'File': path, 'Genes': ';'.join(genes_sorted)})

    print(f"Wrote region {code} ({region_name}) with {len(genes_sorted)} genes -> {path}")

summary_df = pd.DataFrame(rows).sort_values(['Count', 'RegionCode'], ascending=[False, True])
summary_csv = os.path.join(out_dir, 'venn_regions_summary.csv')
summary_df.to_csv(summary_csv, index=False)
print(f"Saved venn regions summary to: {summary_csv}")

# Also create and save the 4-set Venn figure (counts)
try:
    # Ensure top-level graph folder exists
    top_graph = 'graph'
    os.makedirs(top_graph, exist_ok=True)

    plt.figure(figsize=(8, 8))
    venn(dictionary)
    plt.title('4-Set Venn Diagram (counts)')
    figpath = os.path.join(top_graph, 'venn_4set_counts.png')
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 4-set Venn figure (counts) to: {figpath}")
except Exception as e:
    print(f"Failed to create/save 4-set Venn figure: {e}")
