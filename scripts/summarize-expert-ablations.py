#!/usr/bin/env python3
"""Summarize every timing sample; input files remain unchanged."""
import collections,json,pathlib,statistics,sys
rows=collections.defaultdict(list);checks=[]
for arg in sys.argv[1:]:
 for line in pathlib.Path(arg).read_text().splitlines():
  v=json.loads(line)
  if v['type']=='timing':rows[(v['family'],v['m'],v['routing'],v['variant'])].append(v['gpu_ms'])
  elif v['type']=='check':checks.append(v)
result=[dict(zip(['family','m','routing','variant'],key),samples=len(v),median_ms=statistics.median(v),min_ms=min(v),max_ms=max(v)) for key,v in rows.items()]
print(json.dumps({'checks':len(checks),'cpu_reference_points':sum(c['cpu_reference_points'] for c in checks),'max_relative_l2':max((c['relative_l2'] for c in checks),default=0),'timings':result},indent=2))
