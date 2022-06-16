#!/usr/bin/env python

import sys

if len(sys.argv) < 2:
	print("Usage:", sys.argv[0], "<columns> [<file>]", file=sys.stderr)
	sys.exit(1)

columns = [[int(k)-1 for k in j.split('-')] if '-' in j else [int(j)-1, int(j)-1] for j in sys.argv[1].split(',')]
f = sys.stdin if len(sys.argv) < 3 else open(sys.argv[2])

for line in f:
	line = [k.strip() for k in line.strip().split('\t')]
	sel = [[line[j] for j in xrange(c[0], c[-1]+1)] for c in columns]
	print("\t".join([k for s in sel for k in s]))

f.close()
