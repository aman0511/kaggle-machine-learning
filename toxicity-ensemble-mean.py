from collections import defaultdict
from glob import glob
import sys

glob_files = sys.argv[1]
loc_outfile = sys.argv[2]

fields = 6

scores = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

with open(loc_outfile, "wb") as outfile:
    for i, glob_file in enumerate( glob(glob_files) ):
        print( "parsing:", glob_file)

        # sort glob_file by first column, ignoring the first line
        lines = open(glob_file).readlines()
        lines = [lines[0]] + sorted(lines[1:])

        for e, line in enumerate(lines):
            if i == 0 and e == 0:
                outfile.write(line.encode())

            if e > 0:
                row = line.strip().split(",")
                for f in range(fields):
                    scores[(e, row[0])][f] += float(row[f+1])

    for j,k in sorted(scores):
        row_scores = [x/(i+1) for x in scores[(j,k)]]

        outfile.write(("%s,%f,%f,%f,%f,%f,%f\n"%(k, *row_scores)).encode())

print("wrote to %s"%loc_outfile)
