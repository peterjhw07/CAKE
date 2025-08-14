import cv2
import numpy as np
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import pandas as pd
import math
import pyautogui

directory = r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Case studies\UV-Vis Enzyme Catalysis\UV-Vis\21040701_Run' # enter file directory
show_first_spec = "Y"  # enter "Y" or "Yes" if you want to see first image, to aid region limit selection
region_limits_pri = (330, 350)  # enter primary (x1, x2, y1, y2) area for pixel abstraction
region_limits_sec = ""  # enter secondary area for pixel abstraction or "" if not required
region_limits_ter = ""  # enter tertiary area for pixel abstraction or "" if not required
exportpath = r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Case studies\UV-Vis Enzyme Catalysis\UV-Vis\21040701_330-3501.xlsx'  # enter Excel path for export

def rect_add(region_limits, col):
    x_diff = abs(region_limits[0] - region_limits[1])
    y_diff = abs(region_limits[2] - region_limits[3])
    rect = patches.Rectangle((region_limits[0], region_limits[2]),
                              x_diff, y_diff, linewidth=1, edgecolor=col, facecolor='none')
    return rect

def get_date_taken(path):
    return Image.open(path)._getexif()[36867]

def get_date_modified(path):
    return os.path.getmtime(path)

# find the location of original t values
def find_exp_t(data, region_limits, out):
    out_calc_inc = np.zeros(len(data))
    index = np.where(data >= region_limits[0] and data <= region_limits[1])
    out_calc_inc[it] = out[index]
    return out_calc_inc

files = os.listdir(directory)
it_num = len(files)

total = np.empty((it_num, 1))
total_time_data = []
it = 0

if region_limits_sec == "":
    region_limits_sec = region_limits_pri
if region_limits_ter == "":
    region_limits_ter = region_limits_pri

if "Y" in show_first_spec:
    current_file = os.path.join(directory, os.listdir(directory)[0])  # int(it_num/
    df = pd.read_csv(current_file, sep=" ", header=None)
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    # print(df)
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': 15})
    plt.plot(x, y, color='k')
    # plt.title("Raw")
    plt.xlim([min(x), max(x)])
    plt.ylim([min(y), max(y)])
    plt.xlabel("Wavelength / nm")
    plt.ylabel("Absorbance")
    plt.show()

for filename in files:
    with open(os.path.join(directory, filename)) as current_file:
        df = pd.read_csv(current_file, sep=" ", header=None)
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        y_adj = y - np.mean(y[0:9])
        region_limits = region_limits_pri
        index = np.where(np.logical_and(x >= region_limits[0], x <= region_limits[1]))
        x_sect = x[index]
        y_sect = y_adj[index]
        area_sect = scipy.integrate.trapezoid(y_sect, x_sect)
        img_time_data = get_date_modified(current_file.name)

        total_time_data.append(img_time_data)

        total[it] = np.array([area_sect])
        it = it + 1

export_files = np.array(files)
# export_total = np.ndarray.tolist(total)
exportdf = pd.DataFrame({"File": export_files, "Date time": total_time_data, "Area": total[:, 0]})
exportdf.sort_values('Date time')
writer = pd.ExcelWriter(exportpath, engine='openpyxl')
exportdf.to_excel(writer, 'Sheet1', index=False)
writer.save()