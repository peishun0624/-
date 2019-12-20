import sys
import normalize_neologd as nneo
import re
for line in sys.stdin:
    line = line.replace("。","。\n")
    line = re.sub(r'<[^\]]*>', '', line)
    print(nneo.normalize_neologd(line))