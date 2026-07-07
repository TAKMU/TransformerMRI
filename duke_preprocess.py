import pandas as pd
import os.path as path
df = pd.read_csv("./data/metadata.csv")
list = df["Series Description"].unique()
df_protocols = pd.DataFrame({"Protocol": list})
use = df_protocols["Protocol"].isin(["ax dyn 1st pass", "Ph1ax 3d dyn", "Ph1ax 3d dyn MP", "ax 3d dyn 1st pass", "Ph1ax dynamic", "Ph1ax dyn", "Ph1Ax Vibrant MultiPhase", "ax t1 tse", "t1fl3dtradynVIEWSspair 1st pass", "1st ax dyn", "ax dyn 1st pas"])
df_p = df_protocols[use]
df_merge = df.merge(df_p, how = "inner", left_on="Series Description", right_on="Protocol")
print(len(df_merge["Subject ID"].unique()))
df_merge["File Location"] = df_merge['File Location'].apply(lambda p: str(path.join("./data", p.replace("./", ""))))
df_labels = pd.read_csv("./data/duke_data.csv")
print(df_labels.head())
print(df_merge[["Subject ID", "File Location"]])
print(df_labels['Subject ID'].dtype, df_merge['Subject ID'].dtype)
df_final = pd.merge(df_merge,df_labels, on="Subject ID", how="inner")
df_final = df_final[["File Location", "Label"]]
df_final = df_final.rename(columns={"File Location": "image", "Label": "label"})
df_final.to_csv("./data/labels_duke.csv", index=False)
