#!/usr/bin/env python

import sys

if len(sys.argv) < 5:
	print >> sys.stderr, "Usage", sys.argv[0], "<file> <num items>..."
	sys.exit(1)

files = sys.argv[1:len(sys.argv):2]
num_items = [int(sys.argv[i]) for i in xrange(2, len(sys.argv), 2)]

values = set()

with open(files[0]) as f:
	num_items_done = 0
	for line in f:
		if num_items[0] > 0 and num_items_done == num_items[0]:
			break
		num_items_done += 1
		line = [k.strip() for k in line.strip().split("\t")]
		values.add(line[0])

for i in xrange(1, len(files)):
	filename = files[i]
	file_values = set()
	with open(filename) as f:
		num_items_done = 0
		for line in f:
			if num_items[i] > 0 and num_items_done == num_items[i]:
				break
			num_items_done += 1
			line = [k.strip() for k in line.strip().split("\t")]
			file_values.add(line[0])
	new_values = set([value for value in file_values if value in values])
	values = new_values

for value in values:
	print value
