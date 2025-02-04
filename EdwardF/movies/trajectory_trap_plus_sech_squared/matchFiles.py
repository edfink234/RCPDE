import os
import glob
import json
import re

def extract_parts(filename):
  """
  Extracts parts of the form '{integer}_point_{integer}' from the given filename.

  Args:
    filename: The filename string.

  Returns:
    A list of extracted parts.
  """
  pattern = r"(\d+)_point_(\d+)"
  matches = re.findall(pattern, filename)
  parts = ["{}_point_{}".format(match[0], match[1]) for match in matches]
  return parts

def string_to_float(string_value):
    parts = string_value.split("_point_")
    if len(parts) != 2:
      return None  # Invalid format

    integer_part, decimal_part = parts
    return float(integer_part + "." + decimal_part)

def flt_to_str(flt):
    return str(flt).replace(".","_point_")

if __name__=='__main__':
    file_paths = glob.glob("*.*mp4")
    destination_path = "/Users/edwardfinkelstein/AIFeynmanExpressionTrees/trajectory_trap_plus_sech_squared_x_star_0_Omega_0_point_2_x0_min_automated_learning_on_the_fly_test/"
    desired_file_paths = []
    for file in file_paths:
        parts = [string_to_float(part) for part in extract_parts(file)]
        if len(parts) < 3:
            continue
        if 0.66 <= parts[1] <= 0.99 and 0.75 <= parts[2] <= 0.99:
            desired_file_paths.append(file)
    print("\n"*100)
    print(len(desired_file_paths), ' '.join(desired_file_paths), sep='\n')
    os.system(f"cp {' '.join(desired_file_paths)} {destination_path}")
        

