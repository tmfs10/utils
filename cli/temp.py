#!/usr/bin/env python

import sys
import os
import gzip

if len(sys.argv) < 2:
	print >> sys.stderr, "Usage:", sys.argv[0], "<files>"
	sys.exit(1)

m = {}
d = 0
for i in xrange(1, len(sys.argv)):
	with open(sys.argv[i]) as f:
		d2 = 0
		for line in f:
			s = [k.strip() for k in line.strip().split('\t')]
			d2 = len(s)-1
			if s[0] not in m:
				m[s[0]] = [None]*7
			for j in xrange(1, len(s)):
				if s[j] != "X":
					m[s[0]][d+j-1] = s[j]
		d += d2

for g in m:
	print g + "\t" + "\t".join([k if k is not None else "N/A" for k in m[g]])
