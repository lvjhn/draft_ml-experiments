import pandas as pd

class Metadata: 
    def get(context, admin_level):
        basis = f"./data/{context}/metadata/basis/{admin_level}.csv"
        main  = f"./data/{context}/metadata/main/{admin_level}.csv"
        basis_df = pd.read_csv(basis)
        merged_df = basis_df


        if admin_level != "barangays":
            main_df = pd.read_csv(main)
            merged_df = pd.merge(basis_df, main_df, on="Unnamed: 0")

        if admin_level == "provinces":
            province_psgcs = \
                f"./data/{context}/metadata/refs/provinces.psgc.csv"
            province_psgcs_df = \
                pd.read_csv(province_psgcs) 
            merged_df = pd.merge(merged_df, province_psgcs_df, on="Province")

        return merged_df
        