import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import pandas as pd
import math

directory = r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Photos\M.17 RXN Timelapse - DPA CAKE'  # enter file directory
show_first_pic = "Yes"  # enter "Y" or "Yes" if you want to see first image, to aid region limit selection
region_limits_pri = (1611, 1773, 552, 734)  # enter primary (x1, x2, y1, y2) area for pixel abstraction
region_limits_sec = (1651, 1765, 788, 800)  # enter secondary area for pixel abstraction or "" if not required
region_limits_ter = (1663, 1795, 1524, 1533)  # enter tertiary area for pixel abstraction or "" if not required
export_colour = "rgb"  # enter colours to export as 'colour/colours', e.g. 'r' or 'rgb'
exportpath = r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\WM_220317_RGB.xlsx'  # enter Excel path for export

# region_limits_pri = (1264, 1391, 577, 715)  # enter primary (x1, x2, y1, y2) area for pixel abstraction
# region_limits_sec = (1373, 1410, 872, 923)  # enter secondary area for pixel abstraction or "" if not required
# region_limits_ter = (1124, 1199, 784, 814)  # enter tertiary area for pixel abstraction or "" if not required

def rect_add(region_limits, col):
    x_diff = abs(region_limits[0] - region_limits[1])
    y_diff = abs(region_limits[2] - region_limits[3])
    rect = patches.Rectangle((region_limits[0], region_limits[2]),
                              x_diff, y_diff, linewidth=1, edgecolor=col, facecolor='none')
    return rect

def get_date_taken(path):
    return Image.open(path)._getexif()[36867]

def rgb_obtain(region_limits):
    region = img[region_limits[2]:region_limits[3], region_limits[0]:region_limits[1]]
    b, g, r = np.mean(region, axis=(0, 1))
    return (r, g, b)

files = os.listdir(directory)
it_num = len(files)

total = np.empty((it_num, 9))
total_time_data = []
last_digit = np.empty((it_num, 1))
it = 0

if region_limits_sec == "":
    region_limits_sec = region_limits_pri
if region_limits_ter == "":
    region_limits_ter = region_limits_pri

if "Y" in show_first_pic:
    current_img = os.path.join(directory, os.listdir(directory)[int(it_num/2)])  # int(it_num/2)
    img = cv2.imread(current_img)
    plot_col = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(plot_col)
    fig, ax = plt.subplots()
    ax.imshow(plot_col)
    rect_pri = rect_add(region_limits_pri, 'r')
    rect_sec = rect_add(region_limits_sec, 'g')
    rect_ter = rect_add(region_limits_ter, 'b')
    ax.add_patch(rect_ter)
    ax.add_patch(rect_sec)
    ax.add_patch(rect_pri)
    plt.show()

for filename in files:
    with open(os.path.join(directory, filename)) as current_img:
        img = cv2.imread(current_img.name)  # img = Image.open(r'C:\Users\Peter\Desktop\JS_0.36.jpg')

        rgb_pri = rgb_obtain(region_limits_pri)
        rgb_sec = rgb_obtain(region_limits_sec)
        rgb_ter = rgb_obtain(region_limits_ter)
        img_time_data = get_date_taken(current_img.name)
        total_time_data.append(img_time_data)

        total[it] = np.array([rgb_pri[0], rgb_pri[1], rgb_pri[2], rgb_sec[0], rgb_sec[1], rgb_sec[2],
                              rgb_ter[0], rgb_ter[1], rgb_ter[2]])
        it = it + 1

print(total)
export_files = np.array(files)
# export_total = np.ndarray.tolist(total)
exportdf = pd.DataFrame({"File": export_files, "Date time": total_time_data, "Pri_R": total[:, 0], "Sec_R": total[:, 3], "Ter_R": total[:, 6],
                         "Pri_G": total[:, 1], "Sec_G": total[:, 4], "Ter_G": total[:, 7],
                         "Pri_B": total[:, 2], "Sec_B": total[:, 5], "Ter_B": total[:, 8]})
writer = pd.ExcelWriter(exportpath, engine='openpyxl')
exportdf.to_excel(writer, 'x4', index=False)
writer.save()