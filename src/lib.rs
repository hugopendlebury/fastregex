use fancy_regex::Expander;
use fancy_regex::{Captures, Regex, RegexBuilder};
use pyo3::exceptions::PyIndexError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pyfunction;
use pyo3::types::PyDict;
use pyo3::types::PyTuple;
use pyo3::wrap_pyfunction;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;

#[derive(FromPyObject, Clone, Debug)]
enum GroupArgTypes {
    Int(i32),
    Str(String),
}

#[pyclass]
#[derive(Debug, Clone)]
struct Pattern {
    regex: Regex,
    flags: u32,
}

#[pyclass]
#[derive(Debug)]
struct Match {
    #[allow(dead_code)]
    mat: fancy_regex::Match<'static>,
    captures: Captures<'static>,
    named_groups: Vec<Option<String>>,
    text: String,
}

#[pyclass]
struct Scanner {
    // Implement as needed
}

#[pyclass(eq, eq_int)]
#[derive(PartialEq)]
enum RegexFlags {
    NOFLAG = 0,
    IGNORECASE = 1,
    DOTALL = 2,
}

#[pyclass]
struct Constants;

#[pyclass]
struct Sre;

static REGEX_CACHE: OnceLock<Mutex<HashMap<(String, u32), Regex>>> = OnceLock::new();

fn get_regex_cache() -> &'static Mutex<HashMap<(String, u32), Regex>> {
    REGEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[pymethods]
impl Match {
    fn expand(&self, template: &str) -> String {
        let expander = Expander::python();
        expander.expansion(template, &self.captures)
    }

    fn group_zero(&self) -> Option<String> {
        group_int(self, &0)
    }

    #[pyo3(signature = (*args))]
    fn group<'a>(&self, py: Python<'a>, args: Vec<GroupArgTypes>) -> PyResult<Bound<'a, PyAny>> {
        if args.len() == 0 {
            let result = group_int(self, &0);
            match result {
                Some(s) => {
                    let py_str = s.into_pyobject(py)?;
                    Ok(py_str.into_any())
                }
                None => Ok(py.None().into_bound(py)),
            }
        } else {
            if args.len() == 1 {
                let arg = args.get(0).unwrap().clone();
                let result = group_int_name(self, &arg);
                match result {
                    Some(s) => {
                        let py_str = s.into_pyobject(py)?;
                        Ok(py_str.into_any())
                    }
                    None => Err(PyIndexError::new_err(format!("no such group {:?}", arg))),
                }
            } else {
                //Ok(py.None().into_bound(py))

                let mut results: Vec<Bound<'a, PyAny>> = Vec::<Bound<'a, PyAny>>::new();
                for i in 0..args.len() {
                    let arg = args.get(i).unwrap().clone();
                    let result = group_int_name(self, &arg);
                    match result {
                        Some(s) => {
                            let py_str = s.into_pyobject(py)?;
                            results.push(py_str.into_any());
                            //py_str.into_any()
                        }
                        None => results.push(py.None().into_bound(py)),
                    }
                }

                Ok(PyTuple::new(py, results)?.into_any())
            }
        }
    }

    fn groups(&self) -> Vec<Option<String>> {
        self.captures
            .iter()
            .skip(1)
            .map(|m| m.map(|mat| mat.as_str().to_string()))
            .collect()
    }

    fn start(&self, idx: usize) -> Option<usize> {
        self.captures
            .get(idx)
            .map(|m| self.text[..m.start()].chars().count())
    }

    fn end(&self, idx: usize) -> Option<usize> {
        self.captures
            .get(idx)
            .map(|m| self.text[..m.end()].chars().count())
    }

    fn span(&self, idx: usize) -> Option<(usize, usize)> {
        self.captures.get(idx).map(|m| {
            let start = self.text[..m.start()].chars().count();
            let end = self.text[..m.end()].chars().count();
            (start, end)
        })
    }

    fn groupdict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let d = PyDict::new(py);
            self.named_groups.iter().for_each(|gn| {
                if let Some(n) = gn {
                    let named_capture = self.captures.name(n.as_str());
                    if let Some(m) = named_capture {
                        d.set_item(n, m.as_str().to_string()).unwrap();
                    }
                }
            });
            Ok(d.into())
        })
    }
}

fn group_int_name(m: &Match, arg: &GroupArgTypes) -> Option<String> {
    match arg {
        GroupArgTypes::Int(idx) => group_int(m, idx),
        GroupArgTypes::Str(group_name) => group_str(m, group_name),
    }
}

#[pyfunction]
#[pyo3(signature = (pattern, flags=None))]
fn compile(pattern: &str, flags: Option<u32>) -> PyResult<Pattern> {
    let flags = flags.unwrap_or(0);
    let mut cache = get_regex_cache().lock().unwrap();

    if let Some(regex) = cache.get(&(pattern.to_string(), flags)) {
        return Ok(Pattern {
            regex: regex.clone(),
            flags: flags,
        });
    }

    let mut builder = RegexBuilder::new(pattern);

    if flags & 0b0001 != 0 {
        builder.case_insensitive(true);
    }
    /*
    if flags & 0b0010 != 0 {
        builder.multi_line(true);
    }
    if flags & 0b0100 != 0 {
        builder.dot_matches_new_line(true);
    }
    */
    // TODO Add other flags as needed

    let regex = builder
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    cache.insert((pattern.to_string(), flags), regex.clone());
    Ok(Pattern { regex, flags })
}

#[pymethods]
impl Pattern {
    pub fn findall(&self, text: &str) -> PyResult<Vec<String>> {
        findall(self, text)
    }

    pub fn fullmatch(&self, text: &str) -> PyResult<Option<Match>> {
        fullmatch(self, text)
    }

    pub fn flags(&self) -> PyResult<u32> {
        //TODO - Check what flags returns in python
        Ok(self.flags)
    }

    //TODO groupindex
    pub fn r#match(&mut self, text: &str) -> PyResult<Option<Match>> {
        r#match(self, text)
    }

    pub fn search(&self, text: &str) -> PyResult<Option<Match>> {
        search(self, text)
    }

    pub fn split(&self, text: &str) -> PyResult<Vec<String>> {
        split(self, text)
    }

    pub fn sub(&self, repl: &str, text: &str) -> PyResult<String> {
        sub(self, repl, text)
    }

    #[pyo3(signature = (repl, text, count=0))]
    fn subn(&self, repl: &str, text: &str, count: usize) -> PyResult<(String, usize)> {
        subn(self, repl, text, count)
    }

    fn pattern(&self) -> String {
        self.regex.to_string()
    }

    #[getter]
    fn groups(&self) -> usize {
        self.regex.captures_len() - 1
    }
}

#[pyfunction]
fn search(pattern: &Pattern, text: &str) -> PyResult<Option<Match>> {
    let captures = pattern
        .regex
        .captures(text)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

    if let Some(caps) = captures {
        if let Some(mat) = caps.get(0) {
            Ok(Some(Match {
                mat: unsafe { std::mem::transmute(mat) },
                captures: unsafe { std::mem::transmute(caps) },
                named_groups: pattern
                    .regex
                    .capture_names()
                    .map(|name| name.map(|n| n.to_string()))
                    .collect(),
                text: text.to_string(),
            }))
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
}

#[pyfunction(name = "match")]
fn r#match_str(pattern: String, text: &str) -> PyResult<Option<Match>> {
    let regex = Regex::new(&pattern);
    match regex {
        Ok(r) => {
            let mut p = Pattern { regex: r, flags: 0 };
            p.r#match(text)
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "{}",
            e
        ))),
    }
}

#[pyfunction(name = "match")]
fn r#match(pattern: &Pattern, text: &str) -> PyResult<Option<Match>> {
    pattern
        .regex
        .captures(text)
        .and_then(|captures| {
            Ok(if let Some(caps) = captures {
                if let Some(mat) = caps.get(0) {
                    if mat.start() == 0 {
                        Ok(Some(Match {
                            mat: unsafe { std::mem::transmute(mat) },
                            captures: unsafe { std::mem::transmute(caps) },
                            named_groups: pattern
                                .regex
                                .capture_names()
                                .map(|name| name.map(|n| n.to_string()))
                                .collect(),
                            text: text.to_string(),
                        }))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            } else {
                Ok(None)
            })
        })
        .unwrap_or(Ok(None))
}

#[pyfunction]
fn fullmatch(pattern: &Pattern, text: &str) -> PyResult<Option<Match>> {
    pattern
        .regex
        .captures(text)
        .and_then(|captures| {
            Ok(if let Some(caps) = captures {
                if let Some(mat) = caps.get(0) {
                    if mat.as_str() == text {
                        Ok(Some(Match {
                            mat: unsafe { std::mem::transmute(mat) },
                            captures: unsafe { std::mem::transmute(caps) },
                            named_groups: pattern
                                .regex
                                .capture_names()
                                .map(|name| name.map(|n| n.to_string()))
                                .collect(),
                            text: text.to_string(),
                        }))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            } else {
                Ok(None)
            })
        })
        .unwrap_or(Ok(None))
}

#[pyfunction]
fn findall(pattern: &Pattern, text: &str) -> PyResult<Vec<String>> {
    let matches = pattern
        .regex
        .find_iter(text)
        .map(|mat| {
            let res = mat
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?
                .as_str()
                .to_string();
            Ok::<String, PyErr>(res)
        })
        .collect::<Result<Vec<String>, _>>()?;

    Ok(matches)
}

/*
#[pyfunction]
fn finditer(pattern: &Pattern, text: &str) -> PyResult<Vec<Match>> {
    let mut matches: Vec<Match> = Vec::new();
    for result in pattern.regex.captures_iter(text) {
        let caps = result.map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e))
        })?;
        // For each match, push a Match struct for every group from 1 to caps.len()
        for idx in 1..caps.len() {
            if let Some(mat) = caps.get(idx) {
                let static_mat: fancy_regex::Match<'static> = unsafe { std::mem::transmute(mat) };
                let static_caps: Captures<'static> = unsafe { std::mem::transmute(&caps) };
                matches.push(Match {
                    mat: static_mat,
                    captures: static_caps,
                    text: text.to_string(),
                });
            }
        }
    }
    Ok(matches)
}
*/
#[pyfunction]
fn sub(pattern: &Pattern, repl: &str, text: &str) -> PyResult<String> {
    let expander = Expander::python();
    Ok(pattern
        .regex
        .replace_all(text, |caps: &Captures| expander.expansion(repl, caps))
        .into_owned())
}

#[pyfunction]
#[pyo3(signature = (pattern, repl, text, count=0))]
fn subn(pattern: &Pattern, repl: &str, text: &str, count: usize) -> PyResult<(String, usize)> {
    log::info!("count is {}", count);
    let expander = Expander::python();
    let mut replacement_groups = usize::default();
    let result: Result<std::borrow::Cow<'_, str>, fancy_regex::Error> =
        pattern.regex.try_replacen(text, count, |caps: &Captures| {
            let expansion = expander.expansion(repl, caps);
            replacement_groups += 1;
            expansion
        });
    Ok((result.unwrap().to_string(), replacement_groups))
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
fn split(pattern: &Pattern, text: &str) -> PyResult<Vec<String>> {
    let results: Result<Vec<_>, _> = pattern.regex.split(text).collect::<Result<Vec<_>, _>>();

    match results {
        Ok(result) => {
            let parts = result.into_iter().map(String::from).collect();
            Ok(parts)
        }
        Err(err) => Err(PyValueError::new_err(err.to_string())),
    }
}

fn group_str(m: &Match, name: &String) -> Option<String> {
    let named_capture = m.captures.name(name.as_str());
    if let Some(m) = named_capture {
        Some(m.as_str().to_string())
    } else {
        None
    }
}

fn group_int(m: &Match, idx: &i32) -> Option<String> {
    m.captures
        .get((*idx as i32).try_into().unwrap())
        .map(|m| m.as_str().to_string())
}

#[pymodule]
fn fastre(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<Pattern>()?;
    m.add_class::<Match>()?;
    m.add_class::<Scanner>()?;
    m.add_class::<RegexFlags>()?;
    m.add_class::<Constants>()?;
    m.add_class::<Sre>()?;
    //m.add("__version__", "0.2.9")?;
    m.add("__doc__", "")?;
    m.add("__name__", "fastre")?;
    m.add("__package__", "fastre")?;
    m.add(
        "__all__",
        vec![
            "compile",
            "search",
            "match",
            "fullmatch",
            "split",
            "findall",
            //"finditer",
            "sub",
            "subn",
            "escape",
            "purge",
        ],
    )?;

    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add_function(wrap_pyfunction!(r#match, m)?)?;
    m.add_function(wrap_pyfunction!(r#match_str, m)?)?;
    m.add_function(wrap_pyfunction!(fullmatch, m)?)?;
    m.add_function(wrap_pyfunction!(split, m)?)?;
    m.add_function(wrap_pyfunction!(findall, m)?)?;
    //m.add_function(wrap_pyfunction!(finditer, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(subn, m)?)?;
    m.add_function(wrap_pyfunction!(escape, m)?)?;
    m.add_function(wrap_pyfunction!(purge, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::prelude::*;

    // Helper function for Python initialization - not needed in most tests
    fn _setup_python() {
        pyo3::prepare_freethreaded_python();
    }

    #[test]
    fn test_compile_basic() {
        let pattern = compile(r"\d+", None).unwrap();
        assert_eq!(pattern.flags, 0);
        assert_eq!(pattern.regex.as_str(), r"\d+");
    }

    #[test]
    fn test_compile_with_flags() {
        let pattern = compile(r"[a-z]+", Some(1)).unwrap(); // IGNORECASE flag
        assert_eq!(pattern.flags, 1);
    }

    #[test]
    fn test_compile_invalid_pattern() {
        let result = compile(r"[", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_found() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = search(&pattern, "abc123def").unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();
        assert_eq!(match_obj.group_zero(), Some("123".to_string()));
    }

    #[test]
    fn test_search_not_found() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = search(&pattern, "abcdef").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_match_at_start() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = r#match(&pattern, "123abc").unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();
        assert_eq!(match_obj.group_zero(), Some("123".to_string()));
    }

    #[test]
    fn test_match_not_at_start() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = r#match(&pattern, "abc123").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_fullmatch_exact() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = fullmatch(&pattern, "123").unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();
        assert_eq!(match_obj.group_zero(), Some("123".to_string()));
    }

    #[test]
    fn test_fullmatch_partial() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = fullmatch(&pattern, "123abc").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_findall_multiple_matches() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = findall(&pattern, "abc123def456ghi").unwrap();
        assert_eq!(result, vec!["123", "456"]);
    }

    #[test]
    fn test_findall_no_matches() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = findall(&pattern, "abcdef").unwrap();
        assert_eq!(result, Vec::<String>::new());
    }

    #[test]
    fn test_sub_replacement() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = sub(&pattern, "X", "abc123def456").unwrap();
        assert_eq!(result, "abcXdefX");
    }

    #[test]
    fn test_sub_replacement_numbers_groups() {
        let pattern = compile(r"(\w+) (\w+)", None).unwrap();
        let result = sub(&pattern, r"\2, \1 \2.", "James Bond").unwrap();
        assert_eq!(result, "Bond, James Bond.");
    }

    #[test]
    fn test_sub_replacement_named_groups() {
        let pattern = compile(r"(?P<first>\w+) (?P<second>\w+)", None).unwrap();
        let result = sub(&pattern, r"\g<second>, \g<first> \g<second>.", "James Bond").unwrap();
        assert_eq!(result, "Bond, James Bond.");
    }

    #[test]
    fn test_sub_replacement_named_groups_and_positional() {
        let pattern = compile(r"(?P<first>\w+) (?P<second>\w+)", None).unwrap();
        let result = sub(&pattern, r"\g<second>, \1 \2.", "James Bond").unwrap();
        assert_eq!(result, "Bond, James Bond.");
    }

    #[test]
    fn test_subn_replacement_with_count() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = subn(&pattern, "X", "abc123def456", 0).unwrap();
        assert_eq!(result.0, "abcXdefX");
    }

    #[test]
    fn test_subn_replacement_no_count_all_matches() {
        let pattern = compile(r"(\d{2})/(\d{2})/(\d{4})", None).unwrap();
        let text = "Events: 12/25/2023, 01/15/2024, 07/04/2023, 01/07/2027, 01/07/2027, 01/07/2027";
        let result = subn(&pattern, r"\3-\2-\1", text, 0).unwrap();
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
        let result = subn(&pattern, r"\3-\2-\1", text, 0).unwrap();
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
        let result = subn(&pattern, r"\3-\2-\1", text, 4).unwrap();
        assert_eq!(
            result.0,
            "Events: 2023-25-12, 2024-15-01, 2023-04-07, 2027-07-01, 01/07/2027, 01/07/2027"
        );
        assert_eq!(result.1, 4);
    }

    #[test]
    fn test_split_basic() {
        let pattern = compile(r"\s+", None).unwrap();
        let result = split(&pattern, "hello world test").unwrap();
        assert_eq!(result, vec!["hello", "world", "test"]);
    }

    #[test]
    fn test_split_no_matches() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = split(&pattern, "abcdef").unwrap();
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
        let result = search(&pattern, "abc123-456def").unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();
        let groups = match_obj.groups();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0], Some("123".to_string()));
        assert_eq!(groups[1], Some("456".to_string()));
    }

    #[test]
    fn test_match_span() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = search(&pattern, "abc123def").unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();
        let span = match_obj.span(0);
        assert_eq!(span, Some((3, 6))); // Characters 3-6 (123)
    }

    #[test]
    fn test_match_start_end() {
        let pattern = compile(r"\d+", None).unwrap();
        let result = search(&pattern, "abc123def").unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();
        assert_eq!(match_obj.start(0), Some(3));
        assert_eq!(match_obj.end(0), Some(6));
    }

    #[test]
    fn test_named_groups() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let pattern = compile(r"(?P<year>\d{4})-(?P<month>\d{2})", None).unwrap();
            let result = search(&pattern, "Date: 2023-12-25").unwrap();
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
        let result = search(&pattern, "hello world").unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();
        let group_result = group_str(&match_obj, &"word".to_string());
        assert_eq!(group_result, Some("hello".to_string()));
    }

    #[test]
    fn test_group_by_index() {
        let pattern = compile(r"(\w+)", None).unwrap();
        let result = search(&pattern, "hello world").unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();
        let group_result = group_int(&match_obj, &1);
        assert_eq!(group_result, Some("hello".to_string()));
    }

    #[test]
    fn test_pattern_flags_property() {
        let pattern = compile(r"test", Some(1)).unwrap();
        assert_eq!(pattern.flags().unwrap(), 1);
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

        // Cache should be empty now, but we can't directly test this
        // We can only test that purge doesn't error
    }

    #[test]
    fn test_expand_template() {
        let pattern = compile(r"(\w+)\s+(\w+)", None).unwrap();
        let result = search(&pattern, "hello world").unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();
        let expanded = match_obj.expand(r"\2 \1");
        assert_eq!(expanded, "world hello");
    }

    #[test]
    fn test_case_insensitive_flag() {
        let pattern = compile(r"hello", Some(1)).unwrap(); // IGNORECASE
        let result = search(&pattern, "HELLO world").unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();
        assert_eq!(match_obj.group_zero(), Some("HELLO".to_string()));
    }

    #[test]
    fn test_unicode_support() {
        let pattern = compile(r"[\u{1F600}-\u{1F64F}]", None).unwrap();
        let result = search(&pattern, "Hello 😀 World").unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();
        assert_eq!(match_obj.group_zero(), Some("😀".to_string()));
    }

    #[test]
    fn test_empty_pattern() {
        let pattern = compile(r"", None).unwrap();
        let result = search(&pattern, "test").unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();
        assert_eq!(match_obj.group_zero(), Some("".to_string()));
    }

    #[test]
    fn test_complex_pattern() {
        let pattern = compile(r"(\d{1,3}\.){3}\d{1,3}", None).unwrap();
        let result = search(&pattern, "IP: 192.168.1.1 Gateway").unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();
        assert_eq!(match_obj.group_zero(), Some("192.168.1.1".to_string()));
    }

    #[test]
    fn test_multiple_named_groups() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let pattern = compile(r"(?P<protocol>https?)://(?P<domain>[\w.-]+)", None).unwrap();
            let result = search(&pattern, "Visit https://example.com for more").unwrap();
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
        let result = search(&pattern, "abc123def").unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();
        let groups = match_obj.groups();
        assert_eq!(groups.len(), 0);
    }

    #[test]
    fn test_optional_capture_groups() {
        let pattern = compile(r"(\d+)?-(\d+)", None).unwrap();
        let result = search(&pattern, "abc-456def").unwrap();
        assert!(result.is_some());

        let match_obj = result.unwrap();
        let groups = match_obj.groups();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0], None); // Optional group not matched
        assert_eq!(groups[1], Some("456".to_string()));
    }
}
