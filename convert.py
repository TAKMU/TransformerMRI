# convert.py
import os
from pathlib import Path
import SimpleITK as sitk

def convert_patient(patient_dir):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    p = Path(patient_dir)
    patient = p.parts[2]
    out_dir = Path("./data/nifti") / patient
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = sitk.ImageSeriesReader()      # one reader per call, never shared
    written = []
    try:
        for sid in reader.GetGDCMSeriesIDs(str(p)):
            out_file = out_dir / f"{sid}.nii.gz"
            if not out_file.exists():      # skip already-converted series on reruns
                reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(p), sid))
                sitk.WriteImage(reader.Execute(), str(out_file))
            written.append(str(out_file))
        return patient, (";".join(written) if written else None), None
    except Exception as e:
        return patient, None, f"{type(e).__name__}: {e}"

if __name__ == "__main__":
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    df = pd.read_csv("./data/labels_duke.csv")
    with ProcessPoolExecutor(max_workers=16) as ex:
        results = list(tqdm(ex.map(convert_patient, df["image"]), total=len(df)))
    df["patient"] = [r[0] for r in results]
    df["nifti"] = [r[1] for r in results]
    print([(r[0], r[2]) for r in results if r[2]][:5])
    df.to_csv("./data/labels_duke_nifti.csv", index=False)