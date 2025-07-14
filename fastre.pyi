from typing import List, Optional, Tuple
from typing import overload

class Pattern:
    """A compiled regular expression pattern."""
    regex: Regex
    def __init__(self, regex: Regex): ...

    def split(self, text: str) -> List[str]:
        """Splits the text using the pattern"""
        ...

    def fullmatch(pattern: Pattern, text: str) -> Optional[Match]:
        """Try to apply the pattern to all of the string, returning a match object if the whole string matches."""
        ...

    def findall(pattern: Pattern, text: str) -> List[str]:
        """Find all substrings where the regex pattern matches and return them as a list."""
        ...

    def search(pattern: Pattern, text: str) -> Optional[Match]:
        """Scan through a string, looking for any location where the regex pattern matches."""
        ...

    def match(pattern: Pattern, text: str) -> Optional[Match]:
        """Try to apply the pattern at the start of the string, returning a match object if successful."""
        ...

    def split(pattern: Pattern, text: str) -> List[str]:
        """Split the source string by the occurrences of the pattern."""
        ...

    def sub(pattern: Pattern, repl: str, text: str) -> str:
        """Return the string obtained by replacing the leftmost non-overlapping occurrences of the pattern in the string by the replacement repl."""
        ...

    def subn(pattern: Pattern, repl: str, text: str) -> Tuple[str, int]:
        """Perform the same operation as sub(), but return a tuple (new_string, number_of_subs_made)."""
        ...

    def pattern(self) -> str:
        """Return the pattern / regular expression"""
        ...

    def groups(self) -> List[Optional[str]]:
        """Return a list of all groups matched by the pattern."""
        ...

    #'flags',
    #'groupindex',
    #'groups',
    #'scanner',


class Match:
    """A match object returned by a regex search."""
    mat: regex.Match
    def __init__(self, mat: regex.Match): ...
    
    @overload
    def group(self, group: str | int) -> Optional[str]:
        """Return the string matched by the group name or index"""
        ...
        
    @overload
    def group(self) -> Optional[str]:
        """Return the full match"""
        ...

    @overload
    def group(self, *args: str | int) -> Tuple[str]:
        """Return the full match"""
        ...

    def groups(self) -> List[Optional[str]]:
        """Return a list of all groups matched by the pattern."""
        ...
        
    def groupdict(self) -> dict:
        """Returns a dictionary of named groups and the value"""
        ...

    def start(self, idx: int) -> Optional[int]:
        """Return the starting position of the match."""
        ...
        
    def end(self, idx: int) -> Optional[int]:
        """Return the ending position of the match."""
        ...
        
    def span(self, idx: int) -> Optional[Tuple[int, int]]:
        """Return a tuple containing the (start, end) positions of the match."""
        ...

class Scanner:
    """A scanner object (not implemented)."""
    ...

class RegexFlag:
    """A struct representing regex flags."""
    bits: int
    def __init__(self, bits: int): ...

class Constants:
    """A struct for regex constants (not implemented)."""
    ...

class Sre:
    """A struct for regex engine (not implemented)."""
    ...

def compile(pattern: str, flags: Optional[int] = ...) -> Pattern:
    """Compile a regular expression pattern into a regex object."""
    ...

def search(pattern: Pattern, text: str) -> Optional[Match]:
    """Scan through a string, looking for any location where the regex pattern matches."""
    ...

def match(pattern: Pattern, text: str) -> Optional[Match]:
    """Try to apply the pattern at the start of the string, returning a match object if successful."""
    ...

def split(pattern: Pattern, text: str) -> List[str]:
    """Split the source string by the occurrences of the pattern."""
    ...


def fullmatch(pattern: Pattern, text: str) -> Optional[Match]:
    """Try to apply the pattern to all of the string, returning a match object if the whole string matches."""
    ...

def findall(pattern: Pattern, text: str) -> List[str]:
    """Find all substrings where the regex pattern matches and return them as a list."""
    ...

def finditer(pattern: Pattern, text: str) -> List[Match]:
    """Return an iterator yielding match objects over all non-overlapping matches for the pattern in the string."""
    ...

def sub(pattern: Pattern, repl: str, text: str) -> str:
    """Return the string obtained by replacing the leftmost non-overlapping occurrences of the pattern in the string by the replacement repl."""
    ...

def subn(pattern: Pattern, repl: str, text: str) -> Tuple[str, int]:
    """Perform the same operation as sub(), but return a tuple (new_string, number_of_subs_made)."""
    ...

def escape(text: str) -> str:
    """Escape all non-alphanumeric characters in a string."""
    ...

def purge() -> None:
    """Purge the regex cache (not implemented)."""
    ...

__version__: str
__doc__: str
__name__: str
__package__: str
__all__: List[str]