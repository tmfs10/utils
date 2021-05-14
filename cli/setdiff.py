#!/usr/bin/env python

# Preserves order

import sys
if len(sys.argv) < 5:
	print >> sys.stderr, "Usage:", sys.argv[0], "<file A> <numRecords> [<file B> <numRecords>]..."
	sys.exit(1)

if (len(sys.argv)-1)%2 != 0:
	print >> sys.stderr, "# of files have to be even and have to have # of records after them"
	sys.exit(1)

fileA = sys.argv[1]
num_recordsA = int(sys.argv[2])
diff_files = sys.argv[3:len(sys.argv):2]
diff_num_recs = [int(k) for k in sys.argv[4:len(sys.argv):2]]

listA = []
setA = {}
with open(fileA) as f:
	i = 0
	for line in f:
		i += 1
		if num_recordsA > 0 and i > num_recordsA:
			break
		line = [k.strip() for k in line.strip().split("\t")]
		setA[line[0]] = line[1:]
		listA += [line[0]]

for j in xrange(len(diff_files)):
	fileB = diff_files[j]
	num_recordsB = diff_num_recs[j]
	with open(fileB) as f:
		i = 0
		for line in f:
			i += 1
			if num_recordsB > 0 and i > num_recordsB:
				break
			line = [k.strip() for k in line.strip().split("\t")]
			if line[0] in setA: 
				del setA[line[0]]

for el in listA:
	if el in setA:
		print el + "\t" + "\t".join(setA[el])
