#!/usr/bin/env python

import sys
import random

if len(sys.argv) < 3:
	print >> sys.stderr, "Usage:", sys.argv[0], "<file> <# rows to select>"
	sys.exit(1)

num_rows = int(sys.argv[2])
lines = []
total_weight = 0.
with open(sys.argv[1]) as f:
	for line in f:
		line = line.strip().split("\t")
		assert len(line) == 2
		line[1] = float(line[1])
		total_weight += line[1]
		lines += [line]

lines.sort(key = lambda k : -k[1])

selection = []

for i in xrange(num_rows):
	num = random.uniform(0., total_weight)
	for k in xrange(len(lines)):
		num -= lines[k][1]
		if num <= 0.:
			break
	total_weight -= lines[k][1]
	selection += [lines[k]]
	lines.pop(k)

for line in selection:
	print line[0] + "\t" + str(line[1])
