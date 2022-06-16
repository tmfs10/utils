#!/usr/bin/env python

import sys

if len(sys.argv) < 4:
	print >> sys.stderr, "Usage", sys.argv[0], "<gene name mapping file> <target file> <forward map?> [<replicate symbol>]"
	sys.exit(1)

forward = int(sys.argv[3])
rep_sym = None if len(sys.argv) <= 4 else sys.argv[4]

m = {}
with open(sys.argv[1]) as f:
	for line in f:
		line = [k.strip() for k in line.strip().split("\t")]
		if line[1-forward] not in m:
			m[line[1-forward]] = set()
		m[line[1-forward]].add(line[forward])

new_vals = [{}]
with open(sys.argv[2]) as f:
	index = 0
	for line in f:
		if rep_sym in line:
			new_vals += [{}]
			index += 1
			continue

		line_split = [k.strip() for k in line.strip().split("\t")]
		assert len(line_split) == 2, line + "\n" + str(line_split)
		if line_split[0] not in m:
			continue
		expr = float(line_split[1])
		assert expr >= 0.

		for key in m[line_split[0]]:
			if key not in new_vals[index]:
				new_vals[index][key] = []
			new_vals[index][key] += [expr]

for new_val in new_vals:
	for key in new_val:
		new_val[key] = sum(new_val[key])/float(len(new_val[key]))

for i in xrange(len(new_vals)):
	for (key, value) in new_vals[index].iteritems():
		print key + "\t" + str(value)
