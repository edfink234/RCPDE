import os
import glob
import re
import pandas as pd

def round_filename(filename):
    new_filename = filename
    match = re.findall(r"\_point\_([\d\.]+)", filename)
    for num_str in match:
#      print(num_str)
      num_str = "0."+num_str.replace(".","")
      if len(num_str[2:]) > 4:
        rounded_num = round(float(num_str), 4)
        rounded_num = str(rounded_num).replace("0.","")
        num_str = num_str.replace("0.","")
        new_filename = new_filename.replace(num_str, str(rounded_num))

    return new_filename

def rename_files():
    # Get all files with .pth or .pt extensions
    files = glob.glob("*.pth") + glob.glob("*.pt")
#    print(files)
    for file in files:
        new_name = round_filename(file)
        if file != new_name:  # Only rename if the name changes
            print(f"mv {file} {new_name}\n")

if __name__ == "__main__":
    rename_files()
#    df = pd.read_csv("../dataFiles/ICs.txt", names=("x_0", "A", "b", "m", "Omega", "best_loss", "best_time"))
#    df["x_0"] = [round(i, 4) for i in df["x_0"]]
#    print(df)
#    df.to_csv("../dataFiles/ICsBackup.txt", header=None, index=False)

