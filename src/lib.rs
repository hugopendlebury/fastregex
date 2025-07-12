use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use fancy_regex::{Captures, Regex, RegexBuilder};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;


#[pyclass]
struct Pattern {
    regex: Regex,
    flags: u32
}

#[pyclass]
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
    NOFLAG=0,
    IGNORECASE=1,
    DOTALL=2
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
    fn group(&self, idx: usize) -> Option<String> {
        self.captures.get(idx).map(|m| m.as_str().to_string())
    }

    fn groups(&self) -> Vec<Option<String>> {
        self.captures
            .iter()
            .skip(1)
            .map(|m| m.map(|mat| mat.as_str().to_string()))
            .collect()
    }

    fn start(&self, idx: usize) -> Option<usize> {
        self.captures.get(idx).map(|m| {
            self.text[..m.start()].chars().count()
        })
    }

    fn end(&self, idx: usize) -> Option<usize> {
        self.captures.get(idx).map(|m| {
            self.text[..m.end()].chars().count()
        })
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
                match gn {
                    Some(n) => {
                        let named_capture = self.captures.name(n.as_str());
                        match named_capture {
                            Some(m) => {
                                d.set_item(n, m.as_str().to_string()).unwrap();
                            },
                            None => {}
                        }
                        //d.set_item(n, ).unwrap();
                    }
                    None => {}
                }
            });
            Ok(d.into())
        })
    }
}

#[pyfunction]
#[pyo3(signature = (pattern, flags=None))]
fn compile(pattern: &str, flags: Option<u32>) -> PyResult<Pattern> {
    let flags = flags.unwrap_or(0);
    let mut cache = get_regex_cache().lock().unwrap();
    
    if let Some(regex) = cache.get(&(pattern.to_string(), flags)) {
        return Ok(Pattern { regex: regex.clone(), flags: flags });
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

    /* 
    pub fn finditer(&self, text: &str) -> PyResult<Vec<Match>> {

        finditer(self, text)

    }
    */

    pub fn fullmatch(&self, text: &str) -> PyResult<Option<Match>>  {

        fullmatch(self, text)

    }

    pub fn flags(&self) -> PyResult<u32>  {

        //TODO - Check what flags returns in python
        Ok(self.flags)

    }

    //TODO groupindex
    pub fn r#match(&self, text: &str) -> PyResult<Option<Match>> {
        r#match(self, text)
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
                named_groups: pattern.regex.capture_names().map(|name| {
                    if let Some(n) = name {
                        Some(n.to_string())
                    } else {
                        None
                    }
                }).collect(),
                text: text.to_string(),
            }))
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
}

#[pyfunction]
fn r#match(pattern: &Pattern, text: &str) -> PyResult<Option<Match>> {
    pattern.regex.captures(text).and_then(|captures| {

        Ok(if let Some(caps) = captures {
            if let Some(mat) = caps.get(0) {
                if mat.start() == 0 {
                   Ok(Some(Match {
                        mat: unsafe { std::mem::transmute(mat) },
                        captures: unsafe { std::mem::transmute(caps) },
                        named_groups: pattern.regex.capture_names().map(|name| {
                            if let Some(n) = name {
                                Some(n.to_string())
                            } else {
                                None
                            }
                        }).collect(),
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

    }).unwrap_or(Ok(None))


}

#[pyfunction]
fn fullmatch(pattern: &Pattern, text: &str) -> PyResult<Option<Match>> {

    pattern.regex.captures(text).and_then(|captures| {
        Ok(if let Some(caps) = captures {
            if let Some(mat) = caps.get(0) {
                if mat.as_str() == text {
                   Ok(Some(Match {
                        mat: unsafe { std::mem::transmute(mat) },
                        captures: unsafe { std::mem::transmute(caps) },
                        named_groups: pattern.regex.capture_names().map(|name| {
                            if let Some(n) = name {
                                Some(n.to_string())
                            } else {
                                None
                            }
                        }).collect(),
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

    }).unwrap_or(Ok(None))


}



#[pyfunction]
fn findall(pattern: &Pattern, text: &str) -> PyResult<Vec<String>> {

    let matches = pattern
        .regex
        .find_iter(text)
        .map(|mat| {
            let res = mat.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?.as_str().to_string();
            Ok::<String, PyErr>(res)
        }).collect::<Result<Vec<String>, _>>()?;

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
    Ok(pattern.regex.replace_all(text, repl).into_owned())
}

#[pyfunction]
fn subn(pattern: &Pattern, repl: &str, text: &str) -> PyResult<(String, usize)> {
    let result = pattern.regex.replace_all(text, repl);
    let replaced_text = result.clone().into_owned();
    Ok((replaced_text, result.len()))
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

    let results: Result<Vec<_>, _>  = pattern.regex.split(text) 
                                .collect::<Result<Vec<_>, _>>();

    match results {
        Ok(result ) => { let parts = result.into_iter()
                                    .map(|s| String::from(s))
                                    .collect();
                                Ok(parts)

        }
        Err(err) => Err(PyValueError::new_err(err.to_string()))
    }



}

#[pymodule]
fn fastre(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Pattern>()?;
    m.add_class::<Match>()?;
    m.add_class::<Scanner>()?;
    m.add_class::<RegexFlags>()?;
    m.add_class::<Constants>()?;
    m.add_class::<Sre>()?;
    m.add("__version__", "0.2.8")?;
    m.add(
        "__doc__",
        "",
    )?;
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