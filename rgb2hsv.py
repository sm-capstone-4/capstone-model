import colorsys
import csv

f = open("test.csv", "w", newline="")
wr = csv.writer(f)

hsv_list = []
rgb = [196/255, 163/255, 140/255]
hsv_list.append(colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2]))
print(hsv_list)
hsv_list.append("ㅗ모몸")

wr.writecols(hsv_list)