'''
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''
import sys
from os import listdir
from os.path import isfile, join, isdir

# Function that open a files line per line and return the list of the lines
def ExtractLines(file):
  lines = []
  with open(file) as fp:
     line = fp.readline()
     while line:
       lines.append(line)
       line = fp.readline()
  return lines

# Function to write to a file a list of lines
def WriteToFile(lines, file):
  fp = open(file, "w")
  for line in lines:
    fp.write(line)

# Take a list of lines and check that lines starting with '#include' are sorted
# if the line are sorted it will return an empty list, otherwise it returns a new list where the
# header are sorted
def SortHeader(lines):
  sorted = True
  sorted_lines = []
  for index in range(0,len(lines)):
    line = lines[index]
    sorted_lines.append(line)
    if line.startswith("#include"):
      if index > 0 and sorted_lines[-2].startswith("#include"):
        if line.lower() < sorted_lines[-2].lower():
          sorted = False
          position = index
          while position > 0 and sorted_lines[position-1].lower() > line.lower():
            if not sorted_lines[position-1].startswith("#include"):
              break
            sorted_lines[position] = sorted_lines[position-1]
            position = position-1

          sorted_lines[position] = line

  if sorted:
    return []
  return sorted_lines

# Check a c++ file has its header sorted properly and fix them if needed
def CheckAndFix(file):
  lines = ExtractLines(file)
  sorted_lines = SortHeader(lines)
  if len(sorted_lines) > 0:
    print("File not sorted: ", file)
    WriteToFile(sorted_lines, file)

# Returns the list of c++ file in a folder and its subfolder
def GetCppFiles(path):
  files = []
  for f in listdir(path):
    full_path = join(path, f);
    if isfile(full_path):
      if f.endswith('.hpp') or f.endswith('.cpp') or f.endswith('.h') or f.endswith('.c'):
        files.append(full_path)
    elif isdir(full_path):
      files = files + GetCppFiles(full_path)
  return files


def main():
  files = GetCppFiles(sys.argv[1])

  for file in files:
    lines = CheckAndFix(file)

# python3 engine/build/scripts/sort_header.py folder
if __name__ == '__main__':
  main()
