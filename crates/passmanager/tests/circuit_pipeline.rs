// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use qiskit_circuit::{
    Qubit,
    circuit_data::CircuitData,
    dag_circuit::DAGCircuit,
    operations::{Operation, Param, StandardGate},
};
use qiskit_passmanager::*;
use qiskit_transpiler::passes::run_remove_identity_equiv;

struct RemoveIdentities;
impl StaticPass<DAGCircuit> for RemoveIdentities {
    fn run(
        &self,
        mut ir: Box<DAGCircuit>,
        _ctx: &mut PassContext,
    ) -> anyhow::Result<Box<DAGCircuit>> {
        run_remove_identity_equiv(&mut ir, None, None)?;
        Ok(ir)
    }
}

struct CountRz(&'static str);
impl StaticPass<CircuitData> for CountRz {
    fn run(&self, ir: Box<CircuitData>, ctx: &mut PassContext) -> anyhow::Result<Box<CircuitData>> {
        let count = ir.iter().filter(|inst| inst.op.name() == "rz").count();
        ctx.set(self.0.to_owned(), Box::new(count));
        Ok(ir)
    }
}
impl StaticPass<DAGCircuit> for CountRz {
    fn run(&self, ir: Box<DAGCircuit>, ctx: &mut PassContext) -> anyhow::Result<Box<DAGCircuit>> {
        let count = ir
            .op_nodes(false)
            .filter(|(_, inst)| inst.op.name() == "rz")
            .count();
        ctx.set(self.0.to_owned(), Box::new(count));
        Ok(ir)
    }
}

struct CircuitToDag;
impl StaticPass<CircuitData, DAGCircuit> for CircuitToDag {
    fn run(&self, ir: Box<CircuitData>, _ctx: &mut PassContext) -> anyhow::Result<Box<DAGCircuit>> {
        Ok(Box::new(DAGCircuit::from_circuit_data(
            &ir, false, None, None,
        )?))
    }
}

#[test]
fn simple() {
    let mut pm = PassManager::new();
    pm.try_push_static_pass::<CircuitData, _>(CountRz("circuit"))
        .unwrap();
    pm.try_push_static_pass(CircuitToDag).unwrap();
    pm.try_push_static_pass(RemoveIdentities).unwrap();
    pm.try_push_static_pass::<DAGCircuit, _>(CountRz("dag"))
        .unwrap();

    let num_inst = 5;
    let mut qc = CircuitData::with_capacity(1, 0, num_inst, Param::Float(0.0)).unwrap();
    qc.push_standard_gate(StandardGate::RZ, &[Param::Float(1.0)], &[Qubit(0)])
        .unwrap();
    for _ in 1..num_inst {
        qc.push_standard_gate(StandardGate::RZ, &[Param::Float(0.0)], &[Qubit(0)])
            .unwrap();
    }

    let (_dag, ctx) = pm.run::<CircuitData, DAGCircuit>(qc).unwrap();
    assert_eq!(
        *ctx.get("circuit").unwrap().downcast_ref::<usize>().unwrap(),
        num_inst
    );
    assert_eq!(*ctx.get("dag").unwrap().downcast_ref::<usize>().unwrap(), 1);
}
