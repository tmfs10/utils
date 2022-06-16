#!/usr/bin/env python

import sys

if len(sys.argv) < 3:
	print >> sys.stderr, "Usage:", sys.argv[0], "<files>"
	sys.exit(1)

filenames = sys.argv[1:]
files = [open(filename) for filename in filenames]

records = [{} for i in xrange(len(files))]

while True:
	lines = [f.readline().strip() for f in files]
	isgood = False
	for line in lines:
		if not line:
			isgood = False
		else:
			isgood = True
			break
	if not isgood:
		break

	for i in xrange(len(lines)):
		if not lines[i]:
			continue
		split = [k.strip() for k in lines[i].split("\t")]
		records[i][split[0]] = '\t'.join(split[1:])

lengths = [len(r) for r in records]
min_val = min(lengths)
min_index = lengths.index(min_val)

for key in records[min_index]:
	isgood = True
	for r in records:
		if key not in r:
			isgood = False
			break
	if not isgood:
		continue

	print key + '\t' + '\t'.join((r[key] for r in records))
