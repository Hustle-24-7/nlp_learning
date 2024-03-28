import sys
import opencc
converter = opencc.OpenCC("t2s.json")
f_in = open(sys.argv[0], "r")
for line in f_in.readline():
    line = line.strip()
    line_t2s = converter.convert(line)
    print(line_t2s)