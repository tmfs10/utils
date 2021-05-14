#!/usr/bin/env python

import sys

if len(sys.argv) < 3:
	print >> sys.stderr, "Usage", sys.argv[0], "<keys> <rank file>"
	sys.exit(1)

key_file = sys.argv[1]
rank_file = sys.argv[2]

keys = {}
with open(key_file) as f:
	for line in f:
		keys[line.strip()] = -1

with open(rank_file) as f:
	cur_line = 0
	for line in f:
		cur_line += 1
		line = [k.strip() for k in line.strip().split("\t")]
		if line[0] in keys:
			keys[line[0]] = cur_line

for key in keys:
	if keys[key] == -1:
		continue
	assert keys[key] > 0
	print key + "\t" + str(keys[key])
