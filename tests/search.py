#!/usr/bin/env python3

from itertools import product
from sys import modules
from types import ModuleType
from typing import Dict, Literal

from pytest import mark

@mark.parametrize("arguments, library",
    product(
        (
            {
                "description": "Test for a valid email address",
                "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                "test_string": "example@test.com",
                "expected": True
            },
            {
                "description": "Test for an invalid email address (missing domain)",
                "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                "test_string": "example@.com",
                "expected": False
            },
            {
                "description": "Test for a valid phone number (US format)",
                "pattern": r"^\(\d{3}\) \d{3}-\d{4}$",
                "test_string": "(123) 456-7890",
                "expected": True
            },
            {
                "description": "Test for an invalid phone number (wrong format)",
                "pattern": r"^\(\d{3}\) \d{3}-\d{4}$",
                "test_string": "123-456-7890",
                "expected": False
            },
            {
                "description": "Test for a valid URL",
                "pattern": r"^(https?://)(www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$",
                "test_string": "https://www.example.com/path",
                "expected": True
            },
            {
                "description": "Test for an invalid URL (missing scheme)",
                "pattern": r"^(https?://)(www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$",
                "test_string": "www.example.com/path",
                "expected": False
            },
            {
                "description": "Test for a valid IPv4 address",
                "pattern": r"^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\." \
                           r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\." \
                           r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\." \
                           r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
                "test_string": "192.168.1.1",
                "expected": True
            },
            {
                "description": "Test for an invalid IPv4 address (out of range)",
                "pattern": r"^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\." \
                           r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\." \
                           r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\." \
                           r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
                "test_string": "256.100.50.25",
                "expected": False
            },
            {
                "description": "Test for a valid date (YYYY-MM-DD)",
                "pattern": r"^\d{4}-\d{2}-\d{2}$",
                "test_string": "2023-12-31",
                "expected": True
            },
            {
                "description": "Test for an invalid date (wrong format)",
                "pattern": r"^\d{4}-\d{2}-\d{2}$",
                "test_string": "31/12/2023",
                "expected": False
            },
            {
                "description": "Test for a valid hexadecimal color code",
                "pattern": r"^#(?:[0-9a-fA-F]{3}){1,2}$",
                "test_string": "#1A2B3C",
                "expected": True
            },
            {
                "description": "Test for an invalid hexadecimal color code (wrong format)",
                "pattern": r"^#(?:[0-9a-fA-F]{3}){1,2}$",
                "test_string": "#12345G",
                "expected": False
            },
            {
                "description": 'Test for a valid password (at least 8 characters, 1 uppercase, 1 lowercase, 1 digit)',
                'pattern': r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z\d]{8,}$',
                'test_string': 'Password1',
                'expected': True
            },
            {
                'description': 'Test for an invalid password (too short)',
                'pattern': r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z\d]{8,}$',
                'test_string': 'Pass1',
                'expected': False
            },
            {
               'description': 'Test for a string containing only digits',
               'pattern': r'^\d+$',
               'test_string': '123456789',
               'expected': True
           },
           {
               'description': 'Test for a string containing non-digits',
               'pattern': r'^\d+$',
               'test_string': '123abc456',
               'expected': False
           },
           {
               'description': 'Test for a string with whitespace only',
               'pattern': r'^\s*$',
               'test_string': '',
               'expected': True
           },
           {
               'description': 'Test for a string with non-whitespace characters',
               'pattern': r'^\s*$',
               'test_string': '   a   ',
               'expected': False
           },
           {
               'description': 'Test for matching a word boundary',
               'pattern': r'\bword\b',
               'test_string': 'This is a word in a sentence.',
               'expected': True
           },
           {
               'description': 'Test for not matching a word boundary',
               'pattern': r'\bword\b',
               'test_string': 'This is a worded sentence.',
               'expected': False
           },
           {
               'description': 'Test email',
               'pattern': r"^(?!\.)(?!.*\.\.)([a-z0-9_'+\-\.]*)[a-z0-9_'+\-]@([a-z0-9][a-z0-9\-]*\.)+[a-z]{2,}$",
               'test_string': 'valid@domain.com',
               'expected': True
           },
           {
               
               'description': 'Test backtrack',
               'pattern': r"(?i)(a|b|ab)*(?=c)",
               'test_string': 'ababababababababababababababab',
               'expected': False
           },
        ),
        ("fastre", "re"),
    )
)
def test_regex_compile_and_search(
    arguments: Dict[Literal["description", "expected", "pattern", "test_string"], str],
    library: Literal["fastre", "re"]
) -> None:
    __import__(library)
    module: ModuleType = modules[library]
    pattern: module.Pattern = module.compile(arguments["pattern"])  # type: ignore
    match = module.search(pattern, arguments["test_string"]) is not None
    assert match == arguments["expected"], f"Failed: {arguments['description']}"