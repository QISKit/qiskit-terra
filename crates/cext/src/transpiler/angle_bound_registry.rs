#![allow(clippy::missing_safety_doc)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

use crate::pointers::const_ptr_as_ref;

use qiskit_circuit::PhysicalQubit;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::angle_bound_registry::WrapAngleRegistry;

/// Return codes:
///  0 -> success (a DAG was returned in out_dag)
///  1 -> not found (no wrapper for this name)
/// -1 -> error while executing wrapper (exception)
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_wrap_angle_registry_substitute(
    reg: *const WrapAngleRegistry,
    name: *const c_char,
    angles: *const f64,
    num_angles: u32,
    qubits: *const u32,
    num_qubits: u32,
    out_dag: *mut *mut DAGCircuit,
) -> c_int {
    if reg.is_null() || name.is_null() || out_dag.is_null() {
        return -1;
    }

    let reg = unsafe { const_ptr_as_ref(reg) };

    let cstr = unsafe { CStr::from_ptr(name) };
    let name_str = match cstr.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let angles_slice: &[f64] = if angles.is_null() || num_angles == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(angles, num_angles as usize) }
    };

    let qubits_slice: Vec<PhysicalQubit> = if qubits.is_null() || num_qubits == 0 {
        Vec::new()
    } else {
        let raw = unsafe { std::slice::from_raw_parts(qubits, num_qubits as usize) };
        raw.iter().map(|&x| PhysicalQubit(x)).collect()
    };

    match reg.substitute_angle_bounds(name_str, angles_slice, &qubits_slice) {
        Ok(opt_dag) => {
            if let Some(dag) = opt_dag {
                let boxed = Box::new(dag);
                let raw = Box::into_raw(boxed);
                unsafe {
                    *out_dag = raw;
                }
                0
            } else {
                1
            }
        }
        Err(py_err) => {
            let _ = py_err;
            -1
        }
    }
}

/// Create a new WrapAngleRegistry and return pointer to it.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_wrap_angle_registry_new() -> *mut WrapAngleRegistry {
    Box::into_raw(Box::new(WrapAngleRegistry::new()))
}

#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_wrap_angle_registry_free(reg: *mut WrapAngleRegistry) {
    if reg.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(reg);
    };
}
