
#Demonstate the use of expansion / replacement in strings

import fastregex
text = "Bond, James"
match= fastregex.match(r"(?P<last>[^,\s]+),\s+(?P<first>\S+)")
if match:
    print(match.expand(r"The names \g<last>. \g<first> \g<last>"))   

import fastregex
text = "Bond, James"
match = fastregex.match(r"(?P<last>[^,\s]+),\s+(?P<first>\S+)", text)
if match:
    print(match.expand(r"The names \2. \1 \2"))  