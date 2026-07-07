import pandas as pd

df = pd.read_csv("./data/metadata.csv")
df_protocol = pd.read_csv("./protocols.csv")
df["Series Description"] = df["Series Description"].str.lower()
df_protocol["Protocols"] = df_protocol["Protocols"].str.lower()
df = df.rename(columns = {"Series Description" : "Protocols"})
df = pd.merge(df, df_protocol, on="Protocols")
print(len(df["Subject ID"].unique().tolist()))
locations = df["File Location"].tolist()
df_c = pd.read_csv("./data/clinical.csv")
df_c = df_c[["bcr_patient_barcode", "pathologic_T", "pathologic_N", "pathologic_M"]]
df_c = df_c.rename(columns={"bcr_patient_barcode": "Subject ID"})

df = pd.merge(df, df_c, on="Subject ID" )
pat_T = df["pathologic_T"].tolist()
pat_N = df["pathologic_N"].tolist()
pat_M = df["pathologic_M"].tolist()
import re
t_pat = []
for t in pat_T:
    numbers = re.sub(r'\D', '', t)
    t_pat.append(int(numbers))
df_data = pd.DataFrame({"image": locations, "label": t_pat})
print(df_data.to_dict(orient="records"))


