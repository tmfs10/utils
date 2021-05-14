#!/usr/bin/env python

import sys

if len(sys.argv) < 3:
	print >> sys.stderr, "Usage", sys.argv[0], "<file to filter> <filter keys files>"
	sys.exit(1)

file_to_filter = sys.argv[1]
key_files = []
for x in sys.argv[2:]:
	key_files += [x]

keys = set()
for key_file in key_files:
	with open(key_file) as f:
		for line in f:
			s = line.strip()
			if s == "":
				continue
			keys.add(s)

print >> sys.stderr, len(keys), " keys"

with open(file_to_filter) as f:
	for line in f:
		line_split = [k.strip() for k in line.strip().split("\t")]

		skip = True
		for s in line_split:
			if s in keys:
				skip = False
				break
		if not skip:
			print line

