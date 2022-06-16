#!/usr/bin/env python

"""
Converts names to numbers
"""

import sys

assert False, "Not tested"

if len(sys.argv) < 3:
	print >> sys.stderr, "Usage:", sys.argv[0], "<graph file> <comma-separated column list>"
	sys.exit(1)

cols = [int(k)-1 for k in sys.argv[2].split(",")]
names = {}
with open(sys.argv[1]) as f:
	for line in f:
		line = [k.strip() for k in line.strip().split("\t")]

		for col in cols:
			if col < 0:
				print >> sys.stderr, "col # must be >= 1"
			if col >= len(line):
				continue
			if line[col] not in names:
				num = len(names)
				names[line[col]] = num
				line[col] = num
		print "\t".join((str(k) for k in line))

print "== Mapping =="
for name in names:
	print name + "\t" + str(names[name])
