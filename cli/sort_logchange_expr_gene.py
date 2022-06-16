#!/usr/bin/env python

import sys
import math

if len(sys.argv) < 2:
	print >> sys.stderr, "Usage:", sys.argv[0], "<file>"
	sys.exit(1)

genes = {}
with open(sys.argv[1]) as f:
	for line in f:
		line = [k.strip() for k in line.strip().split("\t")]
		assert line[0] not in genes
		expr = float(line[1])
		genes[line[0]] = (expr, abs(math.log(expr)))

keys = genes.keys()
keys = sorted(keys, key=lambda k : genes[k][1], reverse=True)

for key in keys:
	print key + "\t" + str(genes[key][0])
