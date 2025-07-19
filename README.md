<div align="center">
  

  # fastre: A fast Python Regex Engine with support for fancy features

  ![Star](https://img.shields.io/badge/Please%20Give%20A%20Star%20%E2%AD%90-30323D?style=flat-square)
  ![PyPI - Implementation](https://img.shields.io/pypi/implementation/fastre?style=flat-square)
  ![GitHub Issues](https://img.shields.io/github/issues/hugopendlebury/fastre?style=flat-square)
  ![PyPI - Downloads](https://img.shields.io/pypi/dd/fastre?style=flat-square)
  ![GitHub License](https://img.shields.io/github/license/hugopendlebury/fastre?style=flat-square)
  ![GitHub last commit](https://img.shields.io/github/last-commit/hugopendlebury/fastre?display_timestamp=committer&style=flat-square)

  🚀 Supercharge your Python regex with Rust-powered performance!
</div>

## 🌟 Why fastre ?

The python implemention of regular expressions is typically using the re module. The performance of this module can be slow, for some type of expressions.
This module seeks to make regular expressions in python faster. 

fastre is a powerful Python library that wraps the blazing-fast [Rust fancy-regex crate](https://crates.io/crates/fancy-regex), bringing enhanced speed to your regular expression operations. It's designed to be a drop-in replacement for Python's native `re` module.

fastre is based on a fork of an earlier implementation called flpc, which was archived on github in May 2025. Prior to being archived
fastre had numerous reported issues which were not addressed. This module seeks to address those issues and offer further enhancements.
Whilst flpc offered good performance it was based on a rust create which didn't support features such as lookarouds. As such there were many
instances where it couldn't be used a drop in replacement for the python re module. It also renamed the match function
fmatch, and didn't implement some methods on the Match and Pattern objects.

fastre takes a different approach and uses the rust based fancy-regex create. Which means that fastre supports features such as back referencing and lookarounds. One of the key features is that if a regex is considered to be simple then the function will be delegated to the rust based regex crate which
performs operations in constant time.

If a fancy feature is used then an alternative approach is employed based on parsing the regex, building
an Abstract Syntax Tree (AST) and then compiling this into a using an implemention of a Virtual
Machine to execute the progam.


## 🚀 Quick Start

1. Install fastre:
   ```
   pip install fastre
   ```

2. Use it in your code as shown in the API

## 🔧 API

fastre mirrors the `re` module's API, with a few small exceptions:

- When using `group()` on a match object, always provide an index (e.g., `group(0)` for the entire match)

Common functions include:

- `compile()`
- `search()`
- `findall()`
- `finditer()`
- `split()`
- `sub()`
- `subn()`

## 💡 Pro Tips

- Pre-compile your patterns for faster execution
- Use raw strings (`r''`) for cleaner regex patterns
- Always check if a match is found before accessing groups
- Remember to use `group(0)` to get the entire match

## 🤝 Contributing

We welcome contributions! Whether it's bug reports, feature requests, or code contributions, please feel free to reach out. Check our [contribution guidelines](CONTRIBUTING.md) to get started.

## 📄 License

fastre is open-source software licensed under the MIT license.