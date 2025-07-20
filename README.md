<div align="center">
  

  # fastregex: A fast Python Regex Engine with support for fancy features. Aiming to be a drop in replacement for the python re module

  ![Star](https://img.shields.io/badge/Please%20Give%20A%20Star%20%E2%AD%90-30323D?style=flat-square)
  ![PyPI - Implementation](https://img.shields.io/pypi/implementation/fastregex?style=flat-square)
  ![GitHub Issues](https://img.shields.io/github/issues/hugopendlebury/fastregex?style=flat-square)
  ![PyPI - Downloads](https://img.shields.io/pypi/dd/fastregex?style=flat-square)
  ![GitHub License](https://img.shields.io/github/license/hugopendlebury/fastregex?style=flat-square)
  ![GitHub last commit](https://img.shields.io/github/last-commit/hugopendlebury/fastregex?display_timestamp=committer&style=flat-square)

  🚀 Supercharge your Python regex with Rust-powered performance!
</div>

## 🌟 Why fastregex ?

Fastregex is a python module written in rust. Orignally it was a based on a fork of an archived github project called flpc. However the project had
numerous shortcomings. Many methods weren't implemented, it didn't support certain kinds of regular expressions and due to the way it dealt with an
underlying library had issues with memory. Fastregex is a ground up reimplemention.

The python implemention of regular expressions is typically using the re module. The performance of this module can be slow, for some type of expressions.
This module seeks to make regular expressions in python faster. 

fastregex is a powerful Python library that wraps the blazing-fast [Rust fancy-regex crate](https://crates.io/crates/fancy-regex), bringing enhanced speed to your regular expression operations. It's designed to be a drop-in replacement for Python's native `re` module.


fastregex uses the rust based fancy-regex create. Which means that fastregex supports features such as back referencing and lookarounds. One of the key features is that if a regex is considered to be simple then the function will be delegated to the rust based regex crate which
performs operations in constant time.

If a fancy feature is used then an alternative approach is employed based on parsing the regex, building
an Abstract Syntax Tree (AST) and then compiling this into a using an implemention of a Virtual
Machine to execute the progam.


## 🚀 Quick Start

1. Install fastregex:
   ```
   pip install fastregex
   ```

2. Use it in your code as shown in the API

## 🔧 API

fastregex mirrors the `re` module's API.

## 💡 Pro Tips

- Pre-compile your patterns for faster execution
- Use raw strings (`r''`) for cleaner regex patterns
- Always check if a match is found before accessing groups
- Remember to use `group(0)` to get the entire match

## 🤝 Contributing

We welcome contributions! Whether it's bug reports, feature requests, or code contributions, please feel free to reach out. Check our [contribution guidelines](CONTRIBUTING.md) to get started.

## 📄 License

fastregex is open-source software licensed under the MIT license.