#!/usr/bin/env python

"""
Remove duplicates from file based on specificied columns
"""

import sys

if len(sys.argv) < 3:
	print "Usage:", sys.argv[0], "<file>[:separator=\\t] <comma-separated columns> \n \
	Note :- if you have a colon in your filepath, you must specify the separator"
	sys.exit(1)

if ':' in sys.argv[1]:
	sep = sys.argv[1].split(':')[-1]
	filename = sys.argv[1][:-1-len(sep)]
else:
	sep = '\t'
	filename = sys.argv[1] 

cols = [int(k)-1 for k in sys.argv[2].split(",")]

key_set = set()
distinct_dupes = set()
num_duplicates = 0
with open(filename) as f:
	line_num = 0
	for line in f:
		line_num += 1
		if len(line.strip()) == 0:
			continue
		line_split = [k.strip() for k in line.strip().split(sep)]
		key = "\t".join(sorted([line_split[k] for k in cols]))
		if key in key_set: 
			num_duplicates += 1
			distinct_dupes.add(key)
		else:
			key_set.add(key)

print num_duplicates, len(distinct_dupes)
