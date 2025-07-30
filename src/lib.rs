use fancy_regex::{Captures, Regex, RegexBuilder};
use fancy_regex::{Expander, Match};
use pyo3::exceptions::{PyValueError, PyIndexError, PyRuntimeError, PyKeyError};
use pyo3::prelude::*;
use pyo3::pyfunction;
use pyo3::types::PyDict;
use pyo3::types::PyTuple;
use pyo3::wrap_pyfunction;
use pyo3::FromPyObject;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;
use thiserror::Error;
use bitflags::{bitflags, bitflags_match};

#[derive(FromPyObject, Clone, Debug)]
enum NumberString {
    USize(usize),
    Str(String),
}

#[derive(FromPyObject)]
enum Replacement {
    String(String),
    Callable(PyObject),
}

#[derive(FromPyObject, Clone, Debug, Copy)]
enum Flags {
    U32(u32),
    RegexFlag(RegexFlags)
}


#[derive(FromPyObject, Clone, Debug)]
enum PatternOrString {
    Str(String),
    Pattern(Pattern),
}

#[pyclass]
#[derive(Debug, Clone)]
struct Pattern {
    regex: Regex,
    flags: RegexFlags,
}

#[pyclass]
#[derive(Debug, Clone)]
struct MatchLazy {
    pattern: Pattern, 
    string: String,
    named_groups: HashMap<String, usize>,
    named_group_indexes: HashMap<usize, String>,
    match_start: usize,
    match_end: usize,
    capture_positions: Vec<Option<(usize, usize)>>,
}

#[pyclass]
struct Scanner {
    // Implement as needed
}


#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegexFlags {
    bits: u32,
}

#[pymethods]
impl RegexFlags {
    #[classattr]
    const NONE: u32 = 0;
    #[classattr] 
    const TEMPLATE: u32 = 1 << 0;
    #[classattr]
    const IGNORECASE: u32 = 1 << 1;
    #[classattr]
    const LOCALE: u32 = 1 << 2;
    #[classattr]
    const MULTILINE: u32 = 1 << 3;
    #[classattr]
    const DOTALL: u32 = 1 << 4;
    #[classattr]
    const UNICODE: u32 = 1 << 5;
    #[classattr]
    const VERBOSE: u32 = 1 << 6;
    #[classattr]
    const DEBUG: u32 = 1 << 7;
    #[classattr]
    const ASCII: u32 = 1 << 8;

    #[new]
    fn new(bits: Option<u32>) -> Self {
        Self { bits: bits.map_or(0, |f| f) }
    }

    fn __or__(&self, other: &Self) -> Self {
        Self { bits: self.bits | other.bits }
    }

    fn __and__(&self, other: &Self) -> Self {
        Self { bits: self.bits & other.bits }
    }

    fn __xor__(&self, other: &Self) -> Self {
        Self { bits: self.bits ^ other.bits }
    }

    fn contains_flag(&self, other: &Self) -> bool {
        (self.bits & other.bits) == other.bits
    }

    fn contains_value(&self, flag: u32) -> bool {
        (self.bits & flag) != 0
    }

    fn __int__(&self) -> u32 {
        self.bits
    }

    fn __repr__(&self) -> String {
        format!("RegexFlags({})", self.bits)
    }
}

#[derive(Error, Debug)]
pub enum RegexError {
    #[error("Regex compilation failed: {0}")]
    CompileError(#[from] Box<fancy_regex::Error>),
    
    #[error("Pattern matching failed: {0}")]
    MatchError(String),
    
    #[error("Invalid group index: {0}")]
    InvalidGroup(usize),
    
    #[error("Invalid group name: {0}")]
    InvalidGroupName(String),
}

impl From<RegexError> for PyErr {
    fn from(err: RegexError) -> Self {
        match err {
            RegexError::CompileError(_) => PyValueError::new_err(format!("{:#?}", err)),
            RegexError::MatchError(_) => PyRuntimeError::new_err(format!("{:#?}", err)),
            RegexError::InvalidGroup(_) => PyIndexError::new_err(format!("{:#?}", err)),
            RegexError::InvalidGroupName(_) => PyKeyError::new_err(format!("{:#?}", err)),
        }
    }
}


static REGEX_CACHE: OnceLock<Mutex<HashMap<(String, u32), Regex>>> = OnceLock::new();

fn get_regex_cache() -> &'static Mutex<HashMap<(String, u32), Regex>> {
    REGEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}


macro_rules! apply_flag {
    ($flags:ident, $builder:ident, {
        $($flag:ident => $method:ident),* $(,)?
    }) => {
        $(
            if $flags.contains_value(RegexFlags::$flag) {
                $builder.$method(true);
            }
        )*
    };
}

fn get_flags(flags: Option<Flags>) -> RegexFlags {
    match flags {
        Some(f) => {
            match f {
                Flags::U32(u) => RegexFlags::new(Some(u)),
                Flags::RegexFlag(regex_flags) =>regex_flags
            }
        }
        None => RegexFlags::new(None)
    }
}

#[pyfunction]
#[pyo3(signature = (pattern, flags=None))]
fn compile(pattern: &str, flags: Option<Flags>) -> Result<Pattern, RegexError> {

    let flags = get_flags(flags);
    let mut cache = get_regex_cache().lock().unwrap();

    if let Some(regex) = cache.get(&(pattern.to_string(), flags.bits)) {
        return Ok(Pattern {
            regex: regex.clone(),
            flags,
        });
    }

    let mut builder = RegexBuilder::new(pattern);

    apply_flag!(flags, builder, {
        IGNORECASE => case_insensitive,
        MULTILINE  => multi_line,
        VERBOSE    => verbose_mode,  
        UNICODE    => unicode_mode,      
        DOTALL     => dot_matches_new_line,

    });


    let regex = builder.build().map_err(Box::new)?; 

    cache.insert((pattern.to_string(), flags.bits), regex.clone());
    Ok(Pattern { regex, flags })
}

#[pymethods]
impl Pattern {

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("<Pattern '{}'>", self.regex.as_str()))
    }

    pub fn findall(&self, text: &str) -> PyResult<Vec<String>> {
        findall(PatternOrString::Pattern(self.clone()), text, None)
    }

    fn finditer(&self, text: &str) -> PyResult<Vec<MatchLazy>> {
        finditer(PatternOrString::Pattern(self.clone()), text, None)
    }

    pub fn fullmatch(&self, text: &str) -> Result<Option<MatchLazy>, RegexError> {
        match_internal(PatternOrString::Pattern(self.clone()), text, true, true, None)
    }

    pub fn flags(&self) -> PyResult<u32> {
        //TODO - Check what flags returns in python
        Ok(self.flags.bits)
    }

    pub fn r#match(&self, text: &str) -> Result<Option<MatchLazy>, RegexError> {
        r#match(PatternOrString::Pattern(self.clone()), text, None)
    }

    pub fn search(&self, text: &str) -> Result<Option<MatchLazy>, RegexError> {
        search(PatternOrString::Pattern(self.clone()), text, None)
    }

    pub fn split(&self, text: &str) -> Result<Vec<String>, RegexError> {
        split(PatternOrString::Pattern(self.clone()), text, None)
    }

    pub fn sub(&self, repl: Replacement, text: &str) -> PyResult<String> {
        sub(PatternOrString::Pattern(self.clone()), repl, text, None)
    }

    #[pyo3(signature = (repl, text, count=0))]
    fn subn(&self, repl: &str, text: &str, count: usize) -> PyResult<(String, usize)> {
        subn(PatternOrString::Pattern(self.clone()), repl, text, count, None)
    }
    
    #[getter]
    fn pattern(&self) -> String {
        self.regex.to_string()
    }

    #[getter]
    fn groups(&self) -> usize {
        self.regex.captures_len() - 1
    }

    //TODO
    /*
    fn expand(&self, template : &str) -> PyResult<String> {
        sub(self, repl, text)
    }
    */
}

#[pyfunction]
fn search(pattern: PatternOrString, text: &str, flags: Option<Flags>) -> Result<Option<MatchLazy>, RegexError> {
    match_internal(pattern, text, false, false, flags)
}

fn create_pattern(pattern: &PatternOrString, flags: Option<Flags>) -> Result<Pattern, RegexError> {
    match pattern {
        PatternOrString::Str(s) => {

            let regex = compile(s, flags);
            match regex {
                Ok(p) => Ok(Pattern { regex: p.regex, flags:  get_flags(flags)}),
                Err(r) => Err(r)
            }
            
        },
        PatternOrString::Pattern(p) => Ok(p.clone()),
    }
}

pub(crate) fn create_match_object(pattern: &Pattern, text: &str, mat: Match, caps: Captures) -> MatchLazy {
    MatchLazy {
        pattern: pattern.clone(),
        string: text.to_string(),
        named_groups: pattern
            .regex
            .capture_names()
            .enumerate()
            .filter_map(|(index, name)| name.map(|n| (n.to_string(), index)))
            .collect(),
        named_group_indexes: pattern
            .regex
            .capture_names()
            .enumerate()
            .filter_map(|(index, name)| name.map(|n| (index, n.to_string())))
            .collect(),
        match_start: mat.start(),
        match_end: mat.end(),
        capture_positions: caps
            .iter()
            .map(|c| c.map(|m| (m.start(), m.end())))
            .collect(),
    }
}

#[pyfunction]
fn finditer(pattern: PatternOrString, text: &str, flags: Option<Flags>) -> PyResult<Vec<MatchLazy>> {
    let pat = create_pattern(&pattern, flags);

    let mut matches = Vec::<MatchLazy>::new();

    if let Ok(p) = pat {
        for captures in p.regex.captures_iter(text).flatten() {
            if let Some(mat) = captures.get(0) {
                matches.push(create_match_object(&p, text, mat, captures));
            }
        }
    }

    Ok(matches)
}

fn matches(pattern: PatternOrString, text: &str, flags: Option<Flags>) -> Vec<MatchLazy> {
    let pat = create_pattern(&pattern, flags);

    let mut matches = Vec::<MatchLazy>::new();

    if let Ok(p) = pat {
        for captures in p.regex.captures_iter(text).flatten() {
            if let Some(mat) = captures.get(0) {
                matches.push(create_match_object(&p, text, mat, captures));
            }
        }
    }

    matches
}

fn should_match(mat: &Match, text: &str, match_start: bool, match_end: bool) -> bool {
    let starts_at_beginning = mat.start() == 0;
    let ends_at_end = mat.end() == text.len();
    
    match (match_start, match_end) {
        (true, true) => starts_at_beginning && ends_at_end,   // fullmatch
        (true, false) => starts_at_beginning,                  // match
        (false, false) => true,                                // search
        (false, true) => unreachable!("Invalid combination"),
    }
}

fn match_internal(
    pattern: PatternOrString,
    text: &str,
    match_start: bool,
    match_end: bool,
    flags: Option<Flags>
) -> Result<Option<MatchLazy>, RegexError> {
    let pattern = create_pattern(&pattern, flags)?;
    let captures = pattern.regex.captures(text).map_err(Box::new)?;
    
    match captures {
        Some(caps) => {
            if let Some(mat) = caps.get(0) {
                if should_match(&mat, text, match_start, match_end) {
                    Ok(Some(create_match_object(&pattern, text, mat, caps)))
                } else {
                    Ok(None)
                }
            } else {
                Ok(None)
            }
        }
        None => Ok(None),
    }
}

#[pyfunction]
pub(crate) fn r#match(pattern: PatternOrString, text: &str, flags: Option<Flags>) -> Result<Option<MatchLazy>, RegexError> {
    match_internal(pattern, text, true, false, flags)
}

#[pyfunction]
pub(crate) fn fullmatch(pattern: PatternOrString, text: &str, flags: Option<Flags>) -> Result<Option<MatchLazy>, RegexError> {
    match_internal(pattern, text, true, true, flags)
}

fn start_end<'a>(
    py: Python<'a>,
    mat: &MatchLazy,
    element: Option<NumberString>,
    match_accessor: impl Fn(&MatchLazy) -> usize,
    capture_position_accessor: impl Fn((usize, usize)) -> usize,
) -> PyResult<Bound<'a, PyAny>> {
    match element {
        Some(args) => match args {
            NumberString::USize(i) => {
                if i == 0 {
                    Ok(match_accessor(mat).into_pyobject(py)?.into_any())
                } else {
                    //get the result from the vector
                    match mat.capture_positions.get(i) {
                        Some(Some(p)) => {
                            Ok(capture_position_accessor(*p).into_pyobject(py)?.into_any())
                        }
                        Some(None) | None => Err(PyIndexError::new_err(format!("no such group {:?}", i))),
                    }
                }
            }
            NumberString::Str(s) => {
                let group_index = mat.named_groups.get(&s);
                match group_index {
                    Some(gi) => start_end(
                        py,
                        mat,
                        Some(NumberString::USize(*gi)),
                        match_accessor,
                        capture_position_accessor,
                    ),
                    None => Err(PyIndexError::new_err(format!("no such group {:?}", s))),
                }
            }
        },
        None => Ok(match_accessor(mat).into_pyobject(py)?.into_any()),
    }
}

#[pymethods]
impl MatchLazy {
    fn expand(&self, template: &str) -> Result<String, PyErr> {
        sub(
            PatternOrString::Pattern(self.pattern.clone()),
            Replacement::String(template.to_string()),
            self.string.as_str(),
            None
        )
    }

    #[getter]
    fn re(&self) -> Option<String> {
        Some(self.pattern.regex.as_str().to_string())
    }

    fn matchlazy_group_int<'a>(
        &self,
        py: Python<'a>,
        idx: usize,
    ) -> Result<Bound<'a, PyAny>, PyErr> {
        let capture_position = self.capture_positions.get(idx);

        match capture_position {
            Some(ocp) => match ocp {
                Some(span) => {
                    let obj = self.string[span.0..span.1].to_string().into_pyobject(py)?;
                    Ok(obj.into_any())
                }
                None => Ok(py.None().into_pyobject(py)?.into_any()),
            },
            None => Err(PyIndexError::new_err("no such group")),
        }
    }

    fn matchlazy_group_string<'a>(
        &self,
        py: Python<'a>,
        group_name: String,
    ) -> Result<Bound<'a, PyAny>, PyErr> {
        let index = self.named_groups.get(&group_name);
        match index {
            Some(i) => self.matchlazy_group_int(py, *i),
            None => Err(PyIndexError::new_err("no such group")),
        }
    }

    fn group_to_pyobject<'a>(
        &self,
        py: Python<'a>,
        group_value: Option<String>,
    ) -> Result<Bound<'a, PyAny>, PyErr> {
        match group_value {
            Some(s) => {
                let py_str = s.into_pyobject(py)?;
                Ok(py_str.into_any())
            }
            None => Err(PyIndexError::new_err("no such group")),
        }
    }

    #[pyo3(signature = (default=None))]
    fn groups<'a>(
        &self,
        py: Python<'a>,
        default: Option<&str>,
    ) -> Result<Bound<'a, PyTuple>, PyErr> {
        let mut elements: Vec<pyo3::Bound<'_, pyo3::PyAny>> = Vec::new();

        for cap in self.capture_positions.iter().skip(1) {
            let item = match cap {
                Some(c) => self.string[c.0..c.1]
                    .to_string()
                    .into_pyobject(py)?
                    .into_any(),
                None => match default {
                    Some(val) => val.to_string().into_pyobject(py)?.into_any(),
                    None => py.None().into_pyobject(py)?.into_any(),
                },
            };
            elements.push(item);
        }

        let x = PyTuple::new(py, elements);
        x
    }

    #[pyo3(signature = (*args))]
    fn group<'a>(&self, py: Python<'a>, args: Vec<NumberString>) -> PyResult<Bound<'a, PyAny>> {
        if args.is_empty() {
            self.matchlazy_group_int(py, 0)
        } else if args.len() == 1 {
            let arg = args.first().unwrap().clone();
            match arg {
                NumberString::USize(i) => self.matchlazy_group_int(py, i),
                NumberString::Str(s) => self.matchlazy_group_string(py, s),
            }
        } else {
            let results: PyResult<Vec<Bound<'a, PyAny>>> = (0..args.len())
                .map(|i| {
                    let arg = args.get(i).unwrap().clone();
                    match arg {
                        NumberString::USize(i) => self.matchlazy_group_int(py, i),
                        NumberString::Str(s) => self.matchlazy_group_string(py, s),
                    }
                })
                .collect();

            Ok(PyTuple::new(py, results?)?.into_any())
        }
    }

    fn regs<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        let mut results: Vec<Bound<'a, PyAny>> = Vec::<Bound<'a, PyAny>>::new();

        for cap in self.capture_positions.iter().flatten() {
            let tuple = (cap.0, cap.1).into_pyobject(py)?.into_any();
            results.push(tuple);
        }

        Ok(PyTuple::new(py, results)?.into_any())
    }

    fn groupdict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let d = PyDict::new(py);
            self.named_groups.iter().for_each(|gn| {
                let name = gn.0;
                let match_index = gn.1;
                let obj = self.matchlazy_group_int(py, *match_index);
                let _ = match obj {
                    Ok(o) => d.set_item(name, o.into_any()),
                    Err(e) => Err(PyErr::new::<PyRuntimeError, _>(format!(
                        "{}",
                        e
                    ))),
                };
            });
            Ok(d.into())
        })
    }

    #[pyo3(signature = (element = None))]
    fn start<'a>(
        &self,
        py: Python<'a>,
        element: Option<NumberString>,
    ) -> PyResult<Bound<'a, PyAny>> {
        start_end(py, self, element, |m| m.match_start, |t| t.0)
    }

    #[pyo3(signature = (element = None))]
    fn end<'a>(&self, py: Python<'a>, element: Option<NumberString>) -> PyResult<Bound<'a, PyAny>> {
        start_end(py, self, element, |m| m.match_end, |t| t.1)
    }

    #[pyo3(signature = (element = None))]
    fn span<'a>(
        &self,
        py: Python<'a>,
        element: Option<NumberString>,
    ) -> PyResult<Bound<'a, PyAny>> {
        match element {
            Some(args) => match args {
                NumberString::USize(i) => {
                    if i == 0 {
                        Ok((self.match_start, self.match_end)
                            .into_pyobject(py)?
                            .into_any())
                    } else {
                        //get the result from the vector
                        let positions = self.capture_positions.get(i);
                        match positions {
                            Some(pos) => match pos {
                                Some(p) => Ok(p.into_pyobject(py)?.into_any()),
                                None => {
                                    Err(PyIndexError::new_err(format!("no such group {:?}", i)))
                                }
                            },
                            None => Err(PyIndexError::new_err(format!("no such group {:?}", i))),
                        }
                    }
                }
                NumberString::Str(s) => {
                    let group_index = self.named_groups.get(&s);
                    match group_index {
                        Some(gi) => self.span(py, Some(NumberString::USize(*gi))),
                        None => Err(PyIndexError::new_err(format!("no such group {:?}", s))),
                    }
                }
            },
            None => Ok((self.match_start, self.match_end)
                .into_pyobject(py)?
                .into_any()),
        }
    }

    #[getter]
    fn string(&self) -> String {
        self.string.clone()
    }

    #[getter]
    fn endpos(&self) -> usize {
        self.match_end
    }

    #[getter]
    fn lastgroup(&self) -> Option<&String> {
        let x = self
            .capture_positions
            .iter()
            .skip(1)
            .enumerate()
            .filter_map(|(index, captures)| {
                let captured_name = self.named_group_indexes.get(&(index + 1));
                if captures.is_some() {
                    Some((index + 1, captured_name))
                } else {
                    None
                }
            })
            .rev()
            .take(1)
            .map(|f| f.1)
            .last()
            .flatten();
        x
    }

    #[getter]
    fn lastindex(&self) -> Option<usize> {
        let x = self
            .capture_positions
            .iter()
            .skip(1)
            .enumerate()
            .filter_map(|(index, captures)| {
                let captured_name = self.named_group_indexes.get(&(index + 1));
                if captures.is_some() {
                    Some((index + 1, captured_name))
                } else {
                    None
                }
            })
            .rev()
            .take(1)
            .map(|f| f.0)
            .last();
        x
    }
}

#[pyfunction]
fn findall(pattern: PatternOrString, text: &str, flags: Option<Flags>) -> PyResult<Vec<String>> {
    let pattern = create_pattern(&pattern, flags);

    let matches = pattern
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{}", e)))?
        .regex
        .find_iter(text)
        .map(|mat| {
            let res = mat
                .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{}", e)))?
                .as_str()
                .to_string();
            Ok::<String, PyErr>(res)
        })
        .collect::<Result<Vec<String>, _>>()?;

    Ok(matches)
}

#[pyfunction]
fn sub(pattern_or_string: PatternOrString, repl: Replacement, text: &str, flags: Option<Flags>) -> PyResult<String> {
    let pattern = create_pattern(&pattern_or_string, flags);

    match pattern {
        Ok(p) => {
            match repl {
                Replacement::String(r) => {
                    let expander = Expander::python();
                    Ok(p.regex
                        .replace_all(text, |caps: &Captures| expander.expansion(&r, caps))
                        .into_owned())
                }
                Replacement::Callable(callable) => {
                    Python::with_gil(|py| {
                        let callable_ref = callable.bind(py);
                        if !callable_ref.is_callable() {
                            return Err(pyo3::exceptions::PyTypeError::new_err(
                                "Object is not callable",
                            ));
                        }

                        let matches = matches(pattern_or_string, text, flags);
                        let mut last_end = 0; // Start at beginning of text
                        let mut result = String::with_capacity(text.len());
                        for m in matches.iter() {
                            // Add "I have " (chars 0-7)
                            result.push_str(&text[last_end..m.match_start]);

                            let replacement = callable.call1(py, (m.clone(),))?;
                            let replacement_str = replacement.extract::<String>(py)?;

                            result.push_str(&replacement_str);

                            last_end = m.match_end;
                        }

                        result.push_str(&text[last_end..]);
                        Ok(result)
                    })
                }
            }
        }
        Err(e) => Err(PyErr::new::<PyRuntimeError, _>(format!(
            "{}",
            e
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (pattern, repl, text, count=0, flags=None))]
fn subn(
    pattern: PatternOrString,
    repl: &str,
    text: &str,
    count: usize,
    flags: Option<Flags>
) -> PyResult<(String, usize)> {
    let pattern = create_pattern(&pattern, flags);

    match pattern {
        Ok(pat) => {
            let expander = Expander::python();
            let mut replacement_groups = usize::default();
            let result: Result<std::borrow::Cow<'_, str>, fancy_regex::Error> =
                pat.regex.try_replacen(text, count, |caps: &Captures| {
                    let expansion = expander.expansion(repl, caps);
                    replacement_groups += 1;
                    expansion
                });
            Ok((result.unwrap().to_string(), replacement_groups))
        }
        Err(err) => Err(PyValueError::new_err(err.to_string())),
    }
}

#[pyfunction]
fn escape(text: &str) -> PyResult<String> {
    Ok(fancy_regex::escape(text).to_string())
}

#[pyfunction]
fn purge() -> PyResult<()> {
    get_regex_cache().lock().unwrap().clear();
    Ok(())
}

#[pyfunction]
fn split(pattern: PatternOrString, text: &str, flags: Option<Flags>) -> Result<Vec<String>, RegexError> {
    let pattern = create_pattern(&pattern, flags);


    let results: Result<Vec<_>, _> = pattern?.regex.split(text).collect::<Result<Vec<_>, _>>();
    Ok(results.map_err(Box::new)?.into_iter().map(String::from).collect())      
    
}

#[pymodule]
fn fastregex(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<Pattern>()?;
    m.add_class::<Scanner>()?;
    m.add_class::<RegexFlags>()?;
    m.add_class::<MatchLazy>()?;
    m.add_class::<RegexFlags>()?;
    //m.add("__version__", "0.2.9")?;
    m.add("__doc__", "")?;
    m.add("__name__", "fastregex")?;
    m.add("__package__", "fastregex")?;
    m.add(
        "__all__",
        vec![
            "compile",
            "compile",
            "search",
            "match",
            "fullmatch",
            "split",
            "findall",
            "finditer",
            "sub",
            "subn",
            "escape",
            "purge",
            "RegexFlags",
            "Pattern",
            "IGNORECASE",
            "MULTILINE",
            "DOTALL",
            "UNICODE",
            "ASCII",
            "VERBOSE",
            "DEBUG",
            "LOCALE",
            "TEMPLATE",
            "NONE"
        ],
    )?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add_function(wrap_pyfunction!(r#match, m)?)?;
    m.add_function(wrap_pyfunction!(fullmatch, m)?)?;
    m.add_function(wrap_pyfunction!(split, m)?)?;
    m.add_function(wrap_pyfunction!(findall, m)?)?;
    m.add_function(wrap_pyfunction!(finditer, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(subn, m)?)?;
    m.add_function(wrap_pyfunction!(escape, m)?)?;
    m.add_function(wrap_pyfunction!(purge, m)?)?;
    m.add("IGNORECASE", RegexFlags::IGNORECASE)?;
    m.add("MULTILINE", RegexFlags::MULTILINE)?;
    m.add("DOTALL", RegexFlags::DOTALL)?;
    m.add("UNICODE", RegexFlags::UNICODE)?;
    m.add("ASCII", RegexFlags::ASCII)?;
    m.add("VERBOSE", RegexFlags::VERBOSE)?;
    m.add("DEBUG", RegexFlags::DEBUG)?;
    m.add("LOCALE", RegexFlags::LOCALE)?;
    m.add("TEMPLATE", RegexFlags::TEMPLATE)?;
    m.add("NONE", RegexFlags::NONE)?;

    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;

    // Helper function for Python initialization - not needed in most tests
    #[ctor::ctor]
    fn setup_python() {
        pyo3::prepare_freethreaded_python();
    }

    #[test]
    fn test_compile_basic() {
        let pattern = compile(r"\d+", None).unwrap();
        assert_eq!(pattern.flags, RegexFlags::new(None));
        assert_eq!(pattern.regex.as_str(), r"\d+");
    }

    #[test]
    fn test_compile() {
        let f =RegexFlags::new(Some(RegexFlags::IGNORECASE));
        let flags = Flags::RegexFlag(f);
        let pattern = compile(r"[a-z]+", Some(flags)).unwrap(); // IGNORECASE flag
        assert_eq!(pattern.flags, f);
    }

    #[test]
    fn test_compile_invalid_pattern() {
        let result = compile(r"[", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_not_found() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = search(PatternOrString::Pattern(pattern), "abcdef", None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_match_at_start() {
        let regex_str = PatternOrString::Str(r"\d+".to_string());
        let result = r#match(regex_str, "123abc", None).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_match_not_at_start() {
        let pattern = PatternOrString::Pattern(compile(r"\d+", None).unwrap());
        let result = r#match(pattern, "abc123", None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_fullmatch_partial() {
        let pattern = compile(r"\d+", None).unwrap();

        let result = fullmatch(PatternOrString::Pattern(pattern), "123abc", None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_fullmatch_full() {
        let pattern = compile(r"\d+abc", None).unwrap();

        let result = fullmatch(PatternOrString::Pattern(pattern), "123abc", None).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_findall_multiple_matches() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = findall(PatternOrString::Pattern(pattern), "abc123def456ghi", None).unwrap();
        assert_eq!(result, vec!["123", "456"]);
    }

    #[test]
    fn test_findall_no_matches() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = findall(PatternOrString::Pattern(pattern), "abcdef", None).unwrap();
        assert_eq!(result, Vec::<String>::new());
    }

    #[test]
    fn test_sub_replacement() {
        let pattern = compile(r"\d+", None).unwrap();

        let result = sub(
            PatternOrString::Pattern(pattern),
            Replacement::String("X".to_string()),
            "abc123def456",
            None
        )
        .unwrap();
        assert_eq!(result, "abcXdefX");
    }

    #[test]
    fn test_sub_replacement_single_digit() {
        let pattern = compile(r"\d", None).unwrap();

        let result = sub(
            PatternOrString::Pattern(pattern),
            Replacement::String("X".to_string()),
            "abc123def456",
            None
        )
        .unwrap();
        assert_eq!(result, "abcXXXdefXXX");
    }

    #[test]
    fn test_sub_replacement_numbers_groups() {
        let pattern = compile(r"(\w+) (\w+)", None).unwrap();

        let result = sub(
            PatternOrString::Pattern(pattern),
            Replacement::String(r"\2, \1 \2.".to_string()),
            "James Bond",
            None
        )
        .unwrap();
        assert_eq!(result, "Bond, James Bond.");
    }

    #[test]
    fn test_sub_replacement_named_groups() {
        let pattern = compile(r"(?P<first>\w+) (?P<second>\w+)", None).unwrap();

        let result = sub(
            PatternOrString::Pattern(pattern),
            Replacement::String(r"\g<second>, \g<first> \g<second>.".to_string()),
            "James Bond",
            None
        )
        .unwrap();
        assert_eq!(result, "Bond, James Bond.");
    }

    #[test]
    fn test_sub_replacement_named_groups_and_positional() {
        let pattern = compile(r"(?P<first>\w+) (?P<second>\w+)", None).unwrap();

        let result = sub(
            PatternOrString::Pattern(pattern),
            Replacement::String(r"\g<second>, \1 \2.".to_string()),
            "James Bond",
            None
        )
        .unwrap();
        assert_eq!(result, "Bond, James Bond.");
    }

    #[test]
    fn test_subn_replacement_with_count() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = subn(PatternOrString::Pattern(pattern), "X", "abc123def456", 0, None).unwrap();
        assert_eq!(result.0, "abcXdefX");
    }

    #[test]
    fn test_subn_replacement_no_count_all_matches() {
        let pattern = compile(r"(\d{2})/(\d{2})/(\d{4})", None).unwrap();
        let text = "Events: 12/25/2023, 01/15/2024, 07/04/2023, 01/07/2027, 01/07/2027, 01/07/2027";
        let result = subn(PatternOrString::Pattern(pattern), r"\3-\2-\1", text, 0, None).unwrap();
        assert_eq!(
            result.0,
            "Events: 2023-25-12, 2024-15-01, 2023-04-07, 2027-07-01, 2027-07-01, 2027-07-01"
        );
        assert_eq!(result.1, 6);
    }

    #[test]
    fn test_subn_replacement_no_count_five_matches() {
        let pattern = compile(r"(\d{2})/(\d{2})/(\d{4})(,)", None).unwrap();
        let text = "Events: 12/25/2023, 01/15/2024, 07/04/2023, 01/07/2027, 01/07/2027, 01/07/2027";
        let result = subn(PatternOrString::Pattern(pattern), r"\3-\2-\1", text, 0, None).unwrap();
        assert_eq!(
            result.0,
            "Events: 2023-25-12 2024-15-01 2023-04-07 2027-07-01 2027-07-01 01/07/2027"
        );
        assert_eq!(result.1, 5);
    }

    #[test]
    fn test_subn_replacement_count_four() {
        let pattern = compile(r"(\d{2})/(\d{2})/(\d{4})", None).unwrap();
        let text = "Events: 12/25/2023, 01/15/2024, 07/04/2023, 01/07/2027, 01/07/2027, 01/07/2027";
        let result = subn(PatternOrString::Pattern(pattern), r"\3-\2-\1", text, 4, None).unwrap();
        assert_eq!(
            result.0,
            "Events: 2023-25-12, 2024-15-01, 2023-04-07, 2027-07-01, 01/07/2027, 01/07/2027"
        );
        assert_eq!(result.1, 4);
    }

    #[test]
    fn test_split_basic() {
        let pattern = compile(r"\s+", None).unwrap();
        let result = split(PatternOrString::Pattern(pattern), "hello world test", None).unwrap();
        assert_eq!(result, vec!["hello", "world", "test"]);
    }

    #[test]
    fn test_split_no_matches() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = split(PatternOrString::Pattern(pattern), "abcdef", None).unwrap();
        assert_eq!(result, vec!["abcdef"]);
    }

    #[test]
    fn test_escape_special_chars() {
        let result = escape("a.b*c+d?e").unwrap();
        assert_eq!(result, r"a\.b\*c\+d\?e");
    }

    #[test]
    fn test_groups_capture() {
        let pattern = compile(r"(\d+)-(\d+)", None).unwrap();
        let result = search(PatternOrString::Pattern(pattern), "abc123-456def", None).unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();

        Python::with_gil(|py| {
            let groups = match_obj.groups(py, None).unwrap();
            let tuple = groups.downcast::<pyo3::types::PyTuple>().unwrap();
            assert_eq!(tuple.len(), 2);
            assert_eq!(
                Some(tuple.get_item(0).unwrap().extract::<String>().unwrap()),
                Some("123".to_string())
            );
            assert_eq!(
                Some(tuple.get_item(1).unwrap().extract::<String>().unwrap()),
                Some("456".to_string())
            );
        })
    }

    #[test]
    fn test_match_span() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = search(PatternOrString::Pattern(pattern), "abc123def", None).unwrap();
        assert!(result.is_some());

        Python::with_gil(|py| {
            let match_obj = result.unwrap();
            let span = match_obj.span(py, Some(NumberString::USize(0)));
            let value: (i32, i32) = span.unwrap().extract().unwrap();
            assert_eq!(value, (3, 6)); // Characters 3-6 (123)
        })
    }

    #[test]
    fn test_search_start() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = search(PatternOrString::Pattern(pattern), "abc123def", None).unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();

        Python::with_gil(|py| {
            let start = match_obj.start(py, Some(NumberString::USize(0))).unwrap();
            let value: i32 = start.extract().unwrap();
            assert_eq!(value, 3);
        })
    }

    #[test]
    fn test_search_end() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = search(PatternOrString::Pattern(pattern), "abc123def", None).unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();

        Python::with_gil(|py| {
            let start = match_obj.end(py, Some(NumberString::USize(0))).unwrap();
            let value: i32 = start.extract().unwrap();
            assert_eq!(value, 6);
        })
    }

    #[test]
    fn test_named_groups() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let pattern = compile(r"(?P<year>\d{4})-(?P<month>\d{2})", None).unwrap();
            let result = search(PatternOrString::Pattern(pattern), "Date: 2023-12-25", None).unwrap();
            assert!(result.is_some());

            let match_obj = result.unwrap();

            // Test groupdict functionality

            let groupdict = match_obj.groupdict().unwrap();
            let dict = groupdict
                .bind(py)
                .downcast::<pyo3::types::PyDict>()
                .unwrap();

            assert_eq!(
                dict.get_item("year")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "2023"
            );
            assert_eq!(
                dict.get_item("month")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "12"
            );
        });
    }

    #[test]
    fn test_group_by_name() {
        let pattern = compile(r"(?P<word>\w+)", None).unwrap();
        let result = search(PatternOrString::Pattern(pattern), "hello world", None).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_group_by_index() {
        let pattern = compile(r"(\w+)", None).unwrap();
        let result = search(PatternOrString::Pattern(pattern), "hello world", None).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_pattern_flags_property() {
        let flags = RegexFlags::new(Some(RegexFlags::IGNORECASE));
        let f =  Flags::RegexFlag(flags);
        let pattern = compile(r"test", Some(f)).unwrap();
        assert_eq!(pattern.flags().unwrap(), flags.bits);
    }

    #[test]
    fn test_pattern_groups_property() {
        let pattern = compile(r"(\d+)-(\d+)", None).unwrap();
        assert_eq!(pattern.groups(), 2);
    }

    #[test]
    fn test_pattern_pattern_property() {
        let pattern = compile(r"\d+", None).unwrap();
        assert_eq!(pattern.pattern(), r"\d+");
    }

    #[test]
    fn test_cache_functionality() {
        // Test that same pattern is cached
        let pattern1 = compile(r"\d+", None).unwrap();
        let pattern2 = compile(r"\d+", None).unwrap();

        // They should have the same underlying regex (though we can't directly test this)
        assert_eq!(pattern1.pattern(), pattern2.pattern());
        assert_eq!(pattern1.flags, pattern2.flags);
    }

    #[test]
    fn test_purge_cache() {
        let _ = compile(r"\d+", None).unwrap();
        let result = purge();
        assert!(result.is_ok());
    }

    #[test]
    fn test_expand_template() {
        let pattern = compile(r"(\w+)\s+(\w+)", None).unwrap();
        let result = search(PatternOrString::Pattern(pattern), "hello world", None).unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();

        let expanded = match_obj.expand(r"\2 \1");

        assert_eq!(expanded.unwrap(), "world hello");
    }

    #[test]
    fn test_case_insensitive_flag() {
        let flags = RegexFlags::new(Some(RegexFlags::IGNORECASE));
        let f = Flags::RegexFlag(flags);
        let pattern = compile(r"hello", Some(f)).unwrap(); // IGNORECASE
        let result = search(PatternOrString::Pattern(pattern), "HELLO world", None).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_unicode_support() {
        let pattern = compile(r"[\u{1F600}-\u{1F64F}]", None).unwrap();
        let result = search(PatternOrString::Pattern(pattern), "Hello 😀 World", None).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_empty_pattern() {
        let pattern = compile(r"", None).unwrap();
        let result = search(PatternOrString::Pattern(pattern), "test", None).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_complex_pattern() {
        let pattern = compile(r"(\d{1,3}\.){3}\d{1,3}", None).unwrap();
        let result = search(PatternOrString::Pattern(pattern), "IP: 192.168.1.1 Gateway", None).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_multiple_named_groups() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let pattern = compile(r"(?P<protocol>https?)://(?P<domain>[\w.-]+)", None).unwrap();
            let result = search(
                PatternOrString::Pattern(pattern),
                "Visit https://example.com for more",
                None
            )
            .unwrap();
            assert!(result.is_some());

            let match_obj = result.unwrap();

            let groupdict = match_obj.groupdict().unwrap();
            let dict = groupdict
                .bind(py)
                .downcast::<pyo3::types::PyDict>()
                .unwrap();

            assert_eq!(
                dict.get_item("protocol")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "https"
            );
            assert_eq!(
                dict.get_item("domain")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "example.com"
            );
        });
    }

    #[test]
    fn test_no_capture_groups() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = search(PatternOrString::Pattern(pattern), "abc123def", None).unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();

        Python::with_gil(|py| {
            let groups = match_obj.groups(py, None).unwrap();
            assert_eq!(groups.len(), 0);
        })
    }

    #[test]
    fn test_optional_capture_groups() {
        let pattern = compile(r"(\d+)?-(\d+)", None).unwrap();
        let result = search(PatternOrString::Pattern(pattern), "abc-456def", None).unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();

        Python::with_gil(|py| {
            let groups = match_obj.groups(py, None).unwrap();
            assert_eq!(groups.len(), 2);
            let value: (Option<String>, Option<String>) = groups.extract().unwrap();
            assert_eq!(value.0, None); // Optional group not matched
            assert_eq!(value.1, Some("456".to_string()));
        })
    }

    #[test]
    fn test_match_not_matching_is_none() {
        let p = PatternOrString::Str(String::from("Hello"));
        let m = r#match(p, "Good Bye", None).unwrap();
        assert!(m.is_none())
    }

    #[test]
    fn test_match_start_with_string_pattern_no_args() {
        let p = PatternOrString::Str(String::from("Hello"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.start(py, None);
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: i32 = py_result.extract().unwrap();
            assert_eq!(0, value);
        });
    }

    #[test]
    fn test_match_start_with_string_pattern_int_no_match() {
        let p = PatternOrString::Str(String::from("Hello"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();

        Python::with_gil(|py| {
            let result = m.start(py, Some(NumberString::USize(1)));
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_match_end_with_string_pattern_no_args() {
        let p = PatternOrString::Str(String::from("Hello"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let result = m.end(py, None);
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: i32 = py_result.extract().unwrap();
            assert_eq!(5, value);
        });
    }

    #[test]
    fn test_match_end_with_string_pattern_int_no_match() {
        let p = PatternOrString::Str(String::from("Hello"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.end(py, Some(NumberString::USize(1)));
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_start_with_groups_first_int_group() {
        let p = PatternOrString::Str(String::from(r"(Hello)\s(World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.start(py, Some(NumberString::USize(1)));
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: i32 = py_result.extract().unwrap();
            assert_eq!(0, value);
        });
    }

    #[test]
    fn test_start_with_groups_second_int_group() {
        let p = PatternOrString::Str(String::from(r"(Hello)\s(World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.start(py, Some(NumberString::USize(2)));
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: i32 = py_result.extract().unwrap();
            assert_eq!(6, value);
        });
    }

    #[test]
    fn test_end_with_groups_first_int_group() {
        let p = PatternOrString::Str(String::from(r"(Hello)\s(World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.end(py, Some(NumberString::USize(1)));
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: i32 = py_result.extract().unwrap();
            assert_eq!(5, value);
        });
    }

    #[test]
    fn test_end_with_groups_second_int_group() {
        let p = PatternOrString::Str(String::from(r"(Hello)\s(World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.end(py, Some(NumberString::USize(2)));
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: i32 = py_result.extract().unwrap();
            assert_eq!(11, value);
        });
    }

    #[test]
    fn test_end_with_groups_third_int_group() {
        let p = PatternOrString::Str(String::from(r"(Hello)\s(World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.end(py, Some(NumberString::USize(3)));
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_start_with_groups_first_string_group() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.start(py, Some(NumberString::Str("first".to_string())));
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: i32 = py_result.extract().unwrap();
            assert_eq!(0, value);
        });
    }

    #[test]
    fn test_start_with_groups_second_string_group() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.start(py, Some(NumberString::Str("second".to_string())));
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: i32 = py_result.extract().unwrap();
            assert_eq!(6, value);
        });
    }

    #[test]
    fn test_end_with_groups_first_string_group() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.end(py, Some(NumberString::Str("first".to_string())));
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: i32 = py_result.extract().unwrap();
            assert_eq!(5, value);
        });
    }

    #[test]
    fn test_end_with_groups_second_string_group() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.end(py, Some(NumberString::Str("second".to_string())));
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: i32 = py_result.extract().unwrap();
            assert_eq!(11, value);
        });
    }

    #[test]
    fn test_end_with_groups_third_string_group() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.end(py, Some(NumberString::Str("hugo".to_string())));
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_string() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        assert_eq!("Hello World", m.string())
    }

    #[test]
    fn test_span_with_groups_first_string_group() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.span(py, Some(NumberString::Str("first".to_string())));
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: (i32, i32) = py_result.extract().unwrap();
            assert_eq!((0, 5), value);
        });
    }

    #[test]
    fn test_span_with_groups_second_string_group() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.span(py, Some(NumberString::Str("second".to_string())));
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: (i32, i32) = py_result.extract().unwrap();
            assert_eq!((6, 11), value);
        });
    }

    #[test]
    fn test_span_with_groups_first_int_group() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.span(py, Some(NumberString::USize(1)));
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: (i32, i32) = py_result.extract().unwrap();
            assert_eq!((0, 5), value);
        });
    }

    #[test]
    fn test_span_with_groups_second_int_group() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.span(py, Some(NumberString::USize(2)));
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: (i32, i32) = py_result.extract().unwrap();
            assert_eq!((6, 11), value);
        });
    }

    #[test]
    fn test_span_with_groups_no_arg() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.span(py, None);
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: (i32, i32) = py_result.extract().unwrap();
            assert_eq!((0, 11), value);
        });
    }

    #[test]
    fn test_endpos() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        let result = m.endpos();
        assert_eq!(11, result);
    }

    #[test]
    fn test_match_new_group_no_args() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.group(py, Vec::new());
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: String = py_result.extract().unwrap();
            assert_eq!("Hello World", value);
        });
    }

    #[test]
    fn test_match_new_group_int_arg_group_zero() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.group(py, vec![NumberString::USize(0)]);
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: String = py_result.extract().unwrap();
            assert_eq!("Hello World", value);
        });
    }

    #[test]
    fn test_match_new_group_int_arg_group_one() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.group(py, vec![NumberString::USize(1)]);
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: String = py_result.extract().unwrap();
            assert_eq!("Hello", value);
        });
    }

    #[test]
    fn test_match_new_group_int_arg_group_two() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.group(py, vec![NumberString::USize(2)]);
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: String = py_result.extract().unwrap();
            assert_eq!("World", value);
        });
    }

    #[test]
    fn test_match_new_group_int_arg_group_two_and_one() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.group(py, vec![NumberString::USize(2), NumberString::USize(1)]);
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: (String, String) = py_result.extract().unwrap();
            assert_eq!(("World".to_string(), "Hello".to_string()), value);
        });
    }

    #[test]
    fn test_match_new_group_int_arg_group_invalid_index() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.group(py, vec![NumberString::USize(4)]);
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_match_new_group_int_arg_group_invalid_index_throws_no_such_index() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.group(py, vec![NumberString::USize(4)]);

            match result {
                Ok(_) => panic!("Expected an error, but got Ok"),
                Err(err) => {
                    // Check it's the right exception type
                    assert!(err.is_instance_of::<PyIndexError>(py));

                    // Optionally check the error message
                    let err_msg = err.to_string();
                    assert!(err_msg.contains("IndexError: no such group"));
                }
            }
        });
    }

    #[test]
    fn test_match_new_group_string_arg_group_first() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.group(py, vec![NumberString::Str("first".to_string())]);
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: String = py_result.extract().unwrap();
            assert_eq!("Hello", value);
        });
    }

    #[test]
    fn test_match_new_group_string_arg_group_second() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.group(py, vec![NumberString::Str("second".to_string())]);
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: String = py_result.extract().unwrap();
            assert_eq!("World", value);
        });
    }

    #[test]
    fn test_match_new_group_string_arg_group_second_and_first() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.group(
                py,
                vec![
                    NumberString::Str("second".to_string()),
                    NumberString::Str("first".to_string()),
                ],
            );
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: (String, String) = py_result.extract().unwrap();
            assert_eq!(("World".to_string(), "Hello".to_string()), value);
        });
    }

    #[test]
    fn test_match_new_group_string_int_args() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.group(
                py,
                vec![
                    NumberString::USize(1),
                    NumberString::Str("second".to_string()),
                    NumberString::Str("first".to_string()),
                ],
            );
            assert!(result.is_ok());
            let py_result = result.unwrap();

            let value: (String, String, String) = py_result.extract().unwrap();
            assert_eq!(
                (
                    "Hello".to_string(),
                    "World".to_string(),
                    "Hello".to_string()
                ),
                value
            );
        });
    }

    #[test]
    fn test_groups_all_capturing() {
        let p = PatternOrString::Str(String::from(r"(?P<first>Hello)\s(?P<second>World)"));
        let m = r#match(p, "Hello World", None).unwrap().unwrap();
        Python::with_gil(|py| {
            let result = m.groups(py, None);
            assert!(result.is_ok());
            let py_result = result.unwrap();
            let value: (String, String) = py_result.extract().unwrap();
            assert_eq!(("Hello".to_string(), "World".to_string()), value);
        })
    }

    #[test]
    fn test_groups_one_none_capturing() {
        let p = PatternOrString::Str(String::from(
            r"(?P<first>Hello)\s(?P<second>World)\s(?P<numbers>\d+)?",
        ));
        let m = r#match(p, "Hello World some extra text which isn't numbers", None)
            .unwrap()
            .unwrap();
        Python::with_gil(|py| {
            let result = m.groups(py, None);
            assert!(result.is_ok());
            let py_result = result.unwrap();
            let (first, second, third): (String, String, Option<String>) =
                py_result.extract().unwrap();
            assert_eq!(first, "Hello");
            assert_eq!(second, "World");
            assert!(third.is_none());
        })
    }

    #[test]
    fn test_groups_one_none_capturing_with_default() {
        let p = PatternOrString::Str(String::from(
            r"(?P<first>Hello)\s(?P<second>World)\s(?P<numbers>\d+)?",
        ));
        let m = r#match(p, "Hello World some extra text which isn't numbers", None)
            .unwrap()
            .unwrap();
        Python::with_gil(|py| {
            let result = m.groups(py, Some("N/A"));
            assert!(result.is_ok());
            let py_result = result.unwrap();
            let (first, second, third): (String, String, String) = py_result.extract().unwrap();
            assert_eq!(first, "Hello");
            assert_eq!(second, "World");
            assert_eq!(third, "N/A");
        })
    }

    #[test]
    fn test_last_group_all_caputuring() {
        let p = PatternOrString::Str(String::from(r"(?P<first>\w+) (?P<last>\w+)"));
        let m = r#match(p, "John Doe", None).unwrap().unwrap();

        let last_group = m.lastgroup();

        assert_eq!("last", last_group.unwrap())
    }

    #[test]
    fn test_last_group_middle_non_caputuring() {
        let p = PatternOrString::Str(String::from(
            r"(?P<first>\w+)(?P<middle> \w+)?(?P<last> \w+)",
        ));
        let m = r#match(p, "John Doe", None).unwrap().unwrap();

        let last_group = m.lastgroup();

        assert_eq!("last", last_group.unwrap())
    }

    #[test]
    fn test_last_group_or_group() {
        let p = PatternOrString::Str(String::from(r"(?P<first>\w+)|(?P<last>\w+)"));
        let m = r#match(p, "John Doe", None).unwrap().unwrap();

        let last_group = m.lastgroup();

        assert_eq!("first", last_group.unwrap())
    }

    #[test]
    fn test_last_group_last_group_not_named() {
        let p = PatternOrString::Str(String::from(r"(?P<first>\w+)(\w+)"));
        let m = r#match(p, "John Doe", None).unwrap().unwrap();

        let last_group = m.lastgroup();

        assert!(last_group.is_none())
    }

    #[test]
    fn test_last_index_first_group_match() {
        let p = PatternOrString::Str(String::from(r"(\w+)\s+(\d+)?"));
        let m = r#match(p, "John ", None).unwrap().unwrap();
        let last_index = m.lastindex();

        assert_eq!(1, last_index.unwrap())
    }

    #[test]
    fn test_last_index_all_groups_match() {
        let p = PatternOrString::Str(String::from(r"(\w+)\s+(\d+)?"));
        let m = r#match(p, "John 58", None).unwrap().unwrap();
        let last_index = m.lastindex();

        assert_eq!(2, last_index.unwrap())
    }

    #[test]
    fn test_last_index_no_groups_match() {
        let p = PatternOrString::Str(String::from(r"(\d+)?"));
        let m = r#match(p, "John", None).unwrap().unwrap();
        let last_index = m.lastindex();

        assert!(last_index.is_none())
    }

    #[test]
    fn test_last_index_with_named_groups() {
        let p = PatternOrString::Str(String::from(r"(?P<first>\w+) (?P<last>\w+)"));
        let m = r#match(p, "John Doe", None).unwrap().unwrap();
        let last_index = m.lastindex();

        assert_eq!(2, last_index.unwrap())
    }

    #[test]
    fn test_re_returns_the_expression() {
        let p = PatternOrString::Str(String::from(r"(\d+)?"));
        let m = r#match(p, "John", None).unwrap().unwrap();
        let result = m.re();

        assert_eq!(r"(\d+)?", result.unwrap())
    }

    #[test]
    fn test_find_iter() {
        let pattern = r"\b\w+\b";
        let text = "hugo rocks";

        let results = finditer(PatternOrString::Str(pattern.to_string()), text, None);

        assert!(results.is_ok());

        let vec = results.unwrap();
        assert_eq!(2, vec.len());
    }

    #[test]
    fn test_verbose_regex() {
        let regex_text = r"
        ^                   # Start of string
        (\+1\s)?            # Optional country code +1 followed by space
        \(?                 # Optional opening parenthesis
        (\d{3})             # Area code (3 digits)
        \)?                 # Optional closing parenthesis
        [\s\-]?             # Optional separator (space or dash)
        (\d{3})             # First 3 digits of the phone number
        [\s\-]?             # Optional separator (space or dash)
        (\d{4})             # Last 4 digits of the phone number
        $                   # End of string
        ";
        let flags = RegexFlags::new(Some(RegexFlags::VERBOSE));
        let f = Flags::RegexFlag(flags);
        let reggie = compile(regex_text, Some(f)).unwrap();
        let m = reggie.r#match("+1 (123) 456-7890");
        assert!(m.is_ok());
        println!("{:?}", m);
        let match_obj = m.unwrap().unwrap();
        assert_eq!(17, match_obj.endpos())
    }

}
