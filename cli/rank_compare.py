#!/usr/bin/env python

"""
Prints ranks of items side by side for pairs of files
"""

import sys

if len(sys.argv) < 3:
	print >> sys.stderr, "Usage:", sys.argv[0], "<main file> <other files>"
	sys.exit(1)


order = []
compare_rank_set = {}
with open(sys.argv[1]) as f:
	i = 0
	for line in f:
		i += 1
		line = [k.strip() for k in line.strip().split("\t")]
		item = line[0]
		order += [item]
		compare_rank_set[item] = [str(i)] + ["NP"]*(len(sys.argv)-2)

for file_index in xrange(2, len(sys.argv)):
	with open(sys.argv[file_index]) as f:
		i = 0
		for line in f:
			i += 1
			line = [k.strip() for k in line.strip().split("\t")]
			item = line[0]
			if item in compare_rank_set:
				compare_rank_set[item][file_index-1] = str(i)

for item in order:
	print item + "\t" + "\t".join(compare_rank_set[item])
