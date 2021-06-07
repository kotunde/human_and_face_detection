import os
import re

f = open("info.txt", "w")
for i in os.listdir(os.getcwd()): 
    if i.endswith(".jpg") or i.endswith(".JPG"):
        f.write(os.getcwd() + "\\" + i)
        f.write("\n")
        
f.close()
  