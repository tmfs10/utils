#!/usr/bin/env python

import sys
if len(sys.argv) < 7:
	print "Usage:", sys.argv[0], "<tab separated file 1> <key col> <data col> <tab separated file 2> <key col> <data col> ..."
	sys.exit(1)

assert len(sys.argv) % 2 == 1
filenames = [sys.argv[k] for k in xrange(1, len(sys.argv), 3)]
key_col = [int(sys.argv[k])-1 for k in xrange(2, len(sys.argv), 3)]
data_col = [int(sys.argv[k])-1 for k in xrange(3, len(sys.argv), 3)]

data = {}
header = None
for i in xrange(len(filenames)):
	filename = filenames[i]
	with open(filename) as f:
		for line in f:
			line = [k.strip() for k in line.strip().split("\t")]
			if header is None: header = line[key_col[i]]
			if line[key_col[i]] not in data:
				data[line[key_col[i]]] = ['NA']*len(filenames)
			data[line[key_col[i]]][i] = line[data_col[i]]

print header + "\t" + "\t".join(data[header])
for d in data:
	if d == header:
		continue
	print d + "\t" + "\t".join(data[d])
