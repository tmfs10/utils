#!/usr/bin/env python

"""
If the same key is indexing multiple lines, then this script chooses
the line with the max value for the given col for every key.
"""

import sys

if len(sys.argv) < 4:
	print >> sys.stderr, "Usage:", sys.argv[0], "<file> <key> <col>"
	sys.exit(1)

key = int(sys.argv[2])-1
col = int(sys.argv[3])-1
assert key >= 0
assert col >= 0
m = {}
with open(sys.argv[1]) as f:
	for line in f:
		line = [k.strip() for k in line.strip().split("\t")]
		if key >= len(line) or col >= len(line):
			continue

		before_line = "\t".join(line[:col])
		after_line = "\t".join(line[col+1:])

		if line[key] not in m:
			m[line[key]] = [float(line[col]), before_line, after_line]
		if m[line[key]] < float(line[col]):
			m[line[key]] = [float(line[col]), before_line, after_line]

for (k, v) in m.iteritems():
	s = v[1]+"\t"+str(v[0])+"\t"+v[2]
	s = s.strip()
	print s
