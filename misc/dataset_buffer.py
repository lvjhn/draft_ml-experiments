from loaders.datasets import * 

# --- DATASETS ---  # 

""" 
    load_agnc_dataset()
    load_nac_b_dataset()
    load_nt_dataset()
    load_wnc_dataset()
    load_ncd_r_dataset()
    load_nac_t_dataset()
    load_pfncd_dataset()
"""

print("@ Loading AGNC dataset.")
ds = load_agnc_dataset()

print("@ Loading NAC (B) dataset.")
ds = load_nac_b_dataset()

print("@ Loading NT dataset.")
ds = load_nt_dataset()

print("@ Loading WNC dataset.")
ds = load_wnc_dataset()

print("@ Loading NCD (R) dataset.")
ds = load_ncd_r_dataset()

print("@ Loading NAC (T) dataset.")
ds = load_nac_t_dataset()

print("@ Loading PFNCD dataset.")
ds = load_pfncd_dataset()