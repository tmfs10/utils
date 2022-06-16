#!/usr/bin/env python

import sys
import math

if len(sys.argv) < 5:
	print >> sys.stderr, "Usage", sys.argv[0], "<gene name mapping file> <target file> <forward map?> <in log format>"
	sys.exit(1)

gene_map_file = sys.argv[1]
timeseries_file = sys.argv[2]
forward = int(sys.argv[3])
in_log = int(sys.argv[4]) == 1

gene_map = {}
with open(gene_map_file) as f:
	for line in f:
		line = [k.strip() for k in line.strip().split("\t")]
		if line[1-forward] not in gene_map:
			gene_map[line[1-forward]] = set()
		gene_map[line[1-forward]].add(line[forward])

assert len(gene_map) > 0

gene_timeseries = {}
num_rows_eliminated = 0
with open(timeseries_file) as f:
	for line in f:
		print line.strip('\n')
		break
	for line in f:
		line = [k.strip() for k in line.strip().split('\t')]
		source_gene = line[0]
		series = [float(k) for k in line[1:]]
		assert len(series) > 0

		if source_gene not in gene_map:
			num_rows_eliminated += 1
			continue

		for gene in gene_map[source_gene]:
			if gene not in gene_timeseries:
				gene_timeseries[gene] = []
			gene_timeseries[gene] += [series]

print >> sys.stderr, "Num Rows Eliminated:", num_rows_eliminated
num_rows_merge_eliminated = 0
for gene in gene_timeseries:
	assert len(gene_timeseries[gene]) > 0
	best_series = None
	best_line_change = 0.

	num_rows_merge_eliminated += len(gene_timeseries[gene])-1
	for series in gene_timeseries[gene]:
		assert type(series) is list
		assert len(series) > 0
		if in_log:
			min_el = min((series[i] for i in xrange(len(series))))
			max_change = max((series[i]-min_el for i in xrange(len(series))))
		else:
			min_el = min((series[i] for i in xrange(len(series)) if series[i] > 0.))
			max_change = max((abs(math.log(series[i]/min_el)) for i in xrange(len(series)) if series[i] > 0.))

		if max_change > best_line_change:
			best_series = series
	assert series is not None

	print gene + '\t' + '\t'.join((str(k) for k in series))

print >> sys.stderr, "Num Rows Merge Eliminated:", num_rows_merge_eliminated
