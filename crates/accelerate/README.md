# `qiskit-accelerate`

This crate provides a bits-and-pieces Rust library for small, self-contained functions
that are used by the main Python-space components to accelerate certain tasks.  If you're trying to
speed up one particular Python function by replacing its innards with a Rust one, this is the best
place to put the code.  This is _usually_ the right place to put new Rust/Python code.

The `qiskit-pyext` crate is what actually builds the C extension modules.  Modules in here should define
themselves has being submodules of `qiskit._accelerate`, and then the `qiskit-pyext` crate should bind them
into its `fn _accelerate` when it's making the C extension.
### Adding a module to `_accelerate`
Specifically, adding a rust module `my_module` to Qiskit such that it can be called from within Python involves
the following steps:
1. Add a module function to your to your `my_module.rs` file, it should look similar to this:
  
     ```rust
     #[pymodule]
     pub fn my_module(m: &Bound<PyModule>) -> PyResult<()> {}
     ```
     The body of this function should add each class/method/function that you want to be exposed to python through this module. It should look similar to this:
     ```rust
     pub fn my_module(m: &Bound<PyModule>) -> PyResult<()> {
        m.add_wrapped(wrap_pyfunction!(mymodulefunction))?; // For functions.
        m.add_class::<MyModuleClass>()?; // For classes.
     }
     ```
2. To expose the module add `pub mod my_module` at the header of `crates/accelerate/src/lib.rs`.
   ```rust
   pub mod mymodule;
   ...
3. To `crates/pyext/src/lib.rs` 
   * Add my_module::my_module to `use qiskit_accelerate::{`
   * Add `m.add_wrapped(wrap_pyfunction!(my_module))?;` to `fn _accelerate(m: &Bound<PyModule>) -> PyResult<()> {`
4. To `qiskit/__init__.py` add `sys.modules["qiskit._accelerate.my_module”] = _accelerate.module`
5. Compile, and you should be done. Within Python you can now `import qiskit._accelerate.my_module`
