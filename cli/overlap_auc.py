#!/usr/bin/env python

import sys

if len(sys.argv) < 4:
	print >> sys.stderr, "Usage:", sys.argv[0], "<related items/genes file> <item ranking file> <increment>"
	sys.exit(1)

overlap_with_file = sys.argv[1]
overlap_source_file = sys.argv[2]
increment = int(sys.argv[3])

with_items = set()
with open(overlap_with_file) as f:
	for line in f:
		line = [k.strip() for k in line.strip().split("\t")]
		with_items.add(line[0])

len_source_items = 0
num_intersect = 0
with open(overlap_source_file) as f:
	source_items_diff = set()
	for line in f:
		len_source_items += 1
		line = [k.strip() for k in line.strip().split("\t")]
		num_intersect += 1 if line[0] in with_items else 0
		if len_source_items%increment == 0:
			print str(len_source_items) + "\t" + str(num_intersect)

if len_source_items%increment != 0:
	print str(len_source_items) + "\t" + str(num_intersect)
