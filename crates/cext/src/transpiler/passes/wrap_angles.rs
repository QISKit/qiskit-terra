use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::converters::dag_to_circuit;
use qiskit_circuit::dag_circuit::DAGCircuit;

use qiskit_transpiler::angle_bound_registry::WrapAngleRegistry;
use qiskit_transpiler::passes::run_wrap_angles;
use qiskit_transpiler::target::Target;

/// Run the WrapAngles transpiler pass on a circuit.
///
/// This pass applies angle-bounds substitutions from a target's angle bounds using the provided
/// wrap-angle registry. The pass will scan the circuit and replace gates that violate the target's
/// angle bounds using the registry substitution callbacks. The function modifies `circuit` in place.
///
/// @param circuit A pointer to the circuit to run WrapAngles on. The circuit is changed in-place if
///     substitutions are performed. In case of modifications the original circuit's allocations will be
///     replaced by the converted circuit produced from the modified DAG.
///
/// @param target A pointer to a `Target` describing hardware constraints (angle bounds).
///
/// @param bounds_registry A pointer to a `WrapAngleRegistry` which provides substitution callbacks.
///
/// @return 0 on success; negative values indicate an error.
/// # Safety
/// - `circuit`, `target`, and `bounds_registry` must be valid, non-null, and properly aligned.
/// - `circuit` must point to a valid `CircuitData` instance that can be safely mutated.
/// - Behavior is undefined if the pointers passed are invalid or not properly aligned.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_wrap_angles(
    circuit: *mut CircuitData,
    target: *const Target,
    bounds_registry: *const WrapAngleRegistry,
) -> i32 {
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let target = unsafe { const_ptr_as_ref(target) };
    let registry = unsafe { const_ptr_as_ref(bounds_registry) };

    // Convert circuit to DAG
    let mut dag = match DAGCircuit::from_circuit_data(circuit, false, None, None, None, None) {
        Ok(d) => d,
        Err(_) => return -4,
    };

    // Run the pass; run_wrap_angles returns a PyResult<()>, so map errors to error code.
    if run_wrap_angles(&mut dag, target, registry).is_err() {
        return -5;
    }

    let out_circuit = match dag_to_circuit(&dag, false) {
        Ok(c) => c,
        Err(_) => return -6,
    };

    *circuit = out_circuit;
    0
}