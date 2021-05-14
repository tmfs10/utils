#!/usr/bin/env python

from __future__ import print_function
import sys

m = []
quiet = True
f = sys.stdin
if len(sys.argv) > 1:
	f = open(sys.argv[1])
	quiet = False

for line in f:
	m += [[k.strip() for k in line.strip().split("\t")]]
f.close()

if not quiet:
	print("loaded data", file=sys.stderr)
s = ""
for j in range(len(m[0])):
	if not quiet:
		print("doing column", j, file=sys.stderr)
	s = m[0][j]
	for i in range(1, len(m)):
		s += "\t" + m[i][j]
	print(s)
