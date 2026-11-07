"""The top of a cProfile dump of an explorer's main process, by cumulative and by own time.

    python benchmarks/known/pstats_top.py PROFILE.out OUT.txt [N]
"""
import io
import pstats
import sys

prof, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
buf = io.StringIO()
s = pstats.Stats(prof, stream=buf)
s.sort_stats("cumulative").print_stats(n)
s.sort_stats("tottime").print_stats(n)
open(out, "w").write(buf.getvalue())
print(buf.getvalue()[:200])
