#!/usr/bin/env python

from __future__ import print_function
import sys

if len(sys.argv) < 6:
	print("Usage", sys.argv[0], "<filter key file, multicolumn AND> <key col> <num records> <file to filter> <filter columns>", file=sys.stderr)
	sys.exit(1)

key_cols = [int(k)-1 for k in sys.argv[2].split(",")]
num_key_records = int(sys.argv[3])
filter_columns = [int(k)-1 for k in sys.argv[5].split(",")]
filter_file = sys.argv[4]
assert len(key_cols) == len(filter_columns), str(len(key_cols)) + " == " + str(len(filter_columns))

filter_keys = [set() for k in range(len(filter_columns))]
with open(sys.argv[1]) as f:
	max_key = max(key_cols)
	line_num = 0
	for line in f:
		line_num += 1
		if line_num > num_key_records and num_key_records > 0:
			break
		line = line.strip().split("\t")
		assert len(line) > max_key, str(line) + ",\t" + str(max_key)
		for i in range(len(key_cols)):
			filter_keys[i].add(line[key_cols[i]])

with open(filter_file) as f:
	for line in f:
		line_split = [k.strip() for k in line.split("\t")]
		match = True
		for k in range(len(filter_columns)):
			if filter_columns[k] >= len(line_split):
				match = False
				break
			elif line_split[filter_columns[k]] not in filter_keys[k]:
				match = False
				break
		if match:
			sys.stdout.write(line)
