#!/usr/bin/env python

import sys

if len(sys.argv) < 4:
	print >> sys.stderr, "Usage", sys.argv[0], "<num items from each file> <files...>"
	sys.exit(1)

num_items = int(sys.argv[1])
filenames = sys.argv[2:]
assert len(filenames) >= 2

values = set()
for filename in filenames:
	with open(filename) as f:
		i = 0
		for line in f:
			i += 1
			if i > num_items and num_items > 0:
				break
			line = [k.strip() for k in line.strip().split('\t')]
			values.add(line[0])

for value in values:
	print value
