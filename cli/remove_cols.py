#!/usr/bin/env python

import sys

if len(sys.argv) < 3:
	print "Usage:", sys.argv[0], "<column names file> <target file>"
	sys.exit(1)

col_names = set()
with open(sys.argv[1]) as f:
	for line in f:
		col_names.add(line.strip())

col_names_index = set()
with open(sys.argv[2]) as f:
	for line in f:
		row = []
		line = [k.strip() for k in line.strip().split("\t")]
		for i in xrange(len(line)):
			if line[i] in col_names:
				col_names_index.add(i)
			else:
				row += [line[i]]
		print "\t".join(row)
		break

	for line in f:
		row = []
		line = [k.strip() for k in line.strip().split("\t")]
		for i in xrange(len(line)):
			if i not in col_names_index:
				row += [line[i]]
		print "\t".join(row)
