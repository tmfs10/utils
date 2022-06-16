#!/usr/bin/env python

import sys
import random

if len(sys.argv) < 3:
	print >> sys.stderr, "Usage:", sys.argv[0], "<file> <# rows to select>"
	sys.exit(1)

num_rows = int(sys.argv[2])
lines = []
with open(sys.argv[1]) as f:
	lines = [line.strip() for line in f]

selection = random.sample(lines, num_rows)
for line in selection:
	print line
