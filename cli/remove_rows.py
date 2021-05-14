#!/usr/bin/env python

import sys

if len(sys.argv) < 4:
	print >> sys.stderr, "Usage", sys.argv[0], "<remove keys> <file to prune> <key column>"
	sys.exit(1)

filter_keys = set()
with open(sys.argv[1]) as f:
	for line in f:
		filter_keys.add(line.strip())

with open(sys.argv[2]) as f:
	filter_col=int(sys.argv[3])-1
	for line in f:
		line_split = [k.strip() for k in line.split("\t")]
		if filter_col == -1:
			skip = False
			for el in line_split:
				if el in filter_keys:
					skip = True
					break
			if skip:
				continue
		else:
			if filter_col >= len(line_split):
				continue
			if line_split[filter_col] in filter_keys:
				continue
		print line.strip()
