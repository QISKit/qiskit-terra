================
Dynamic Circuits
================

Dynamic circuits allow measuring qubits during circuit execution and using the measurement results to control subsequent
operations. Qiskit provides extensive support for constructing and reasoning about dynamic circuits. For more details, see the
`Classical feedforward and control flow guide <https://quantum.cloud.ibm.com/docs/guides/classical-feedforward-and-control-flow>`_.

Qiskit's C API currently only supports the inspection of control flow instructions and classical expressions.
Support for constructing classical expressions and adding control flow instructions will be added in future
Qiskit releases.

When working with this API, keep the following assumptions and limitations in mind:

* Most objects returned by the control flow and classical expressions API are borrowed read-only pointers
  (returned as ``const *``). These remain valid only as long as the parent object - for example, the
  circuit that owns an ``IfElse`` instruction - is alive. As such, callers must not free borrowed pointers,
  and must ensure that parent objects outlive any use of those pointers.

* This API does not use error codes. When called correctly, the functions are infallible. However,
  variant-specific functions (e.g. :c:func:`qk_control_flow_box_duration_kind` which expects a ``Box`` instruction) will 
  panic and abort the process used with an object of the wrong type. To guard against this, a set of query functions 
  is provided to check the type or kind of an object before calling the appropriate variant-specific function.

* Qiskit uses big integers to represent some control flow and classical expression constructs,
  such as classical register condition values and switch case labels. Full big integer support
  will be added to the C API in the future. Until then, numerical values in this API are
  limited to what fits in a ``uint64_t``.

The following example program demonstrates all API functions and types for inspecting control flow
instructions and classical expressions, along with selected classical register query functions. The entry
point is ``inspect_circuit`` at the bottom; it calls the helper functions defined above it.

.. code-block:: c

    // Forward declarations for recursive circuit inspection
    void inspect_circuit(const QkCircuit *, const QkControlFlowInstruction *, int);

    void inspect_register(const QkClassicalRegister *creg, int indent) {
        size_t num_bits = qk_classical_register_num_bits(creg);
        char *reg_name = qk_classical_register_name(creg);

        printf("%*sClassical Register: name='%s', num_bits=%zu\n", indent, "",
            reg_name ? reg_name : "<unnamed>", num_bits);

        qk_str_free(reg_name);
    }

    void inspect_expr(const QkExprNode *expr_node, int indent) {
        QkExprNodeKind kind = qk_expr_kind(expr_node);
        printf("%*sExpression kind: %d\n", indent, "", kind);

        switch (kind) {
        case QkExprNodeKind_Unary: {
            QkUnaryExprInfo unary = qk_expr_unary_info(expr_node);
            printf("%*sUnary operation: op=%d, type=%d\n", indent, "", unary.op, unary.ty.ty);
            printf("%*sOperand:\n", indent, "");
            inspect_expr(unary.operand, indent + 2);
            break;
        }
        case QkExprNodeKind_Binary: {
            QkBinaryExprInfo binary = qk_expr_binary_info(expr_node);
            printf("%*sBinary operation: op=%d, type=%d\n", indent, "", binary.op, binary.ty.ty);
            printf("%*sLeft operand:\n", indent, "");
            inspect_expr(binary.left, indent + 2);
            printf("%*sRight operand:\n", indent, "");
            inspect_expr(binary.right, indent + 2);
            break;
        }
        case QkExprNodeKind_Cast: {
            QkCastExprInfo cast = qk_expr_cast_info(expr_node);
            printf("%*sCast to type: %d", indent, "", cast.ty.ty);
            if (cast.ty.ty == QkExprType_Uint) {
                printf(" (width=%u)", cast.ty.width);
            }
            printf("\n%*sOperand:\n", indent, "");
            inspect_expr(cast.operand, indent + 2);
            break;
        }
        case QkExprNodeKind_Index: {
            QkIndexExprInfo index = qk_expr_index_info(expr_node);
            printf("%*sIndex operation, type=%d\n", indent, "", index.ty.ty);
            printf("%*sTarget:\n", indent, "");
            inspect_expr(index.target, indent + 2);
            printf("%*sIndex:\n", indent, "");
            inspect_expr(index.index, indent + 2);
            break;
        }
        case QkExprNodeKind_Value: {
            const QkValue *value = qk_expr_as_value(expr_node);
            QkExprTypeInfo value_type = qk_value_type_info(value);

            printf("%*sValue type: %d", indent, "", value_type.ty);

            switch (value_type.ty) {
            case QkExprType_Duration: {
                QkDurationInfo duration_info = qk_value_duration_info(value);

                if (duration_info.ty == QkDurationType_Dt) {
                    printf(", value=%ld dt\n", duration_info.value.dt);
                } else {
                    printf(", value=%f (unit: %d)\n", duration_info.value.time, duration_info.ty);
                }
                break;
            }
            case QkExprType_Float: {
                double float_val = qk_value_float(value);
                printf(", value=%f\n", float_val);
                break;
            }
            case QkExprType_Uint: {
                uint64_t val = qk_value_uint(value);
                printf(" (width=%u), value=%lu\n", value_type.width, val);
                break;
            }
            case QkExprType_Bool: {
                bool bool_val = qk_value_bool(value);
                printf(", value=%s\n", bool_val ? "true" : "false");
                break;
            }
            }
            break;
        }
        case QkExprNodeKind_Var: {
            const QkVar *var = qk_expr_as_var(expr_node);
            char *name = qk_var_name(var);

            QkExprTypeInfo type_info = qk_var_type_info(var);
            printf("%*sVariable: name='%s', type=%d", indent, "", name ? name : "<unnamed>",
                type_info.ty);
            if (type_info.ty == QkExprType_Uint) {
                printf(" (width=%u)", type_info.width);
            }
            printf("\n");
            if (name != NULL) {
                qk_str_free(name);
            }
            break;
        }
        case QkExprNodeKind_Stretch: {
            const QkStretch *stretch = qk_expr_as_stretch(expr_node);
            char *name = qk_stretch_name(stretch);
            printf("%*sStretch: name='%s'\n", indent, "", name);
            qk_str_free(name);
            break;
        }
        }
    }

    void inspect_condition(const QkControlFlowInstruction *cf_inst, int indent) {
        QkConditionType condition_type = qk_control_flow_condition_type(cf_inst);
        printf("%*sCondition type: %d\n", indent, "", condition_type);

        switch (condition_type) {
        case QkConditionType_ClBit: {
            QkConditionBitInfo cond_bit_info = qk_control_flow_condition_bit_info(cf_inst);
            printf("%*sCondition on classical bit: clbit=%u, value=%s\n", indent, "",
                cond_bit_info.clbit, cond_bit_info.condition ? "true" : "false");
            break;
        }
        case QkConditionType_ClReg: {
            uint64_t cond_width = qk_control_flow_condition_reg_cond_bit_width(cf_inst);
            printf("%*sCondition on classical register (width=%lu bits)\n", indent, "", cond_width);

            if (cond_width <= 64) {
                uint64_t condition = qk_control_flow_condition_reg_cond_uint(cf_inst);
                printf("%*sCondition value: %lu\n", indent, "", condition);
            } else {
                printf("%*sCondition value too large (>64 bits) for direct display\n", indent, "");
            }

            const QkClassicalRegister *creg = qk_control_flow_condition_reg(cf_inst);
            inspect_register(creg, indent + 2);
            break;
        }
        case QkConditionType_Expr: {
            printf("%*sCondition based on expression:\n", indent, "");
            const QkExprNode *expr = qk_control_flow_condition_expr(cf_inst);
            inspect_expr(expr, indent + 2);
            break;
        }
        }
    }

    void inspect_box(const QkControlFlowInstruction *cf_inst, int indent) {
        printf("%*sInspecting Box instruction\n", indent, "");

        QkBoxDurationKind duration_type = qk_control_flow_box_duration_kind(cf_inst);

        switch (duration_type) {
        case QkBoxDurationKind_NoDuration:
            printf("%*sNo duration specified\n", indent, "");
            break;
        case QkBoxDurationKind_Duration: {
            QkDurationInfo duration_info = qk_control_flow_box_duration_val_info(cf_inst);
            printf("%*sDuration: ", indent, "");
            if (duration_info.ty == QkDurationType_Dt) {
                printf("%ld dt\n", duration_info.value.dt);
            } else {
                printf("%f (unit: %d)\n", duration_info.value.time, duration_info.ty);
            }
            break;
        }
        case QkBoxDurationKind_Expr: {
            printf("%*sDuration specified by expression:\n", indent, "");
            const QkExprNode *expr = qk_control_flow_box_duration_expr(cf_inst);
            inspect_expr(expr, indent + 2);
            break;
        }
        }
    }

    void inspect_for_loop(const QkControlFlowInstruction *cf_inst, int indent) {
        printf("%*sInspecting ForLoop instruction\n", indent, "");

        QkLoopCollectionType collection_type = qk_control_flow_loop_collection_type(cf_inst);
        printf("%*sCollection type: %s\n", indent, "",
            collection_type == QkLoopCollectionType_List ? "List" : "Range");

        switch (collection_type) {
        case QkLoopCollectionType_List: {
            QkLoopElements loop_elements = qk_control_flow_loop_elements(cf_inst);
            printf("%*sLoop elements (%zu items): [", indent, "", loop_elements.len);
            for (size_t i = 0; i < loop_elements.len; i++) {
                printf("%zu%s", loop_elements.elements[i], i < loop_elements.len - 1 ? ", " : "");
            }
            printf("]\n");
            break;
        }
        case QkLoopCollectionType_Range: {
            int64_t start, stop, step;
            qk_control_flow_loop_range(cf_inst, &start, &stop, &step);
            printf("%*sLoop range: start=%ld, stop=%ld, step=%ld\n", indent, "", start, stop, step);
            break;
        }
        }

        // Inspect the loop parameter, if it exists
        QkLoopParamKind param_kind = qk_control_flow_loop_param_kind(cf_inst);

        switch (param_kind) {
        case QkLoopParamKind_NoLoopParam:
            printf("%*sNo loop parameter\n", indent, "");
            break;
        case QkLoopParamKind_Parameter: {
            QkSymbolInfo symbol_info = qk_control_flow_loop_symbol_info(cf_inst);
            printf("%*sLoop parameter (Symbol): ", indent, "");
            if (symbol_info.ty == QkSymbolType_Standalone) {
                printf("name='%s'\n", symbol_info.name ? symbol_info.name : "<unnamed>");
            } else if (symbol_info.ty == QkSymbolType_Element) {
                printf("element index=%zu\n", symbol_info.index);
            }
            qk_str_free(symbol_info.name);
            break;
        }
        case QkLoopParamKind_Variable: {
            const QkVar *var = qk_control_flow_loop_variable(cf_inst);
            char *name = qk_var_name(var);
            QkExprTypeInfo type_info = qk_var_type_info(var);
            printf("%*sLoop parameter (Variable): name='%s', type=%d", indent, "",
                name ? name : "<unnamed>", type_info.ty);
            if (type_info.ty == QkExprType_Uint) {
                printf(" (width=%u)", type_info.width);
            }
            printf("\n");
            if (name != NULL) {
                qk_str_free(name);
            }
            break;
        }
        }
    }

    void inspect_switch(const QkControlFlowInstruction *cf_inst, int indent) {
        printf("%*sInspecting Switch instruction\n", indent, "");

        // Inspect the Switch instruction target
        QkConditionType target_type = qk_control_flow_switch_target_type(cf_inst);
        printf("%*sTarget type: %d\n", indent, "", target_type);

        switch (target_type) {
        case QkConditionType_ClBit: {
            uint32_t bit = qk_control_flow_switch_target_bit(cf_inst);
            printf("%*sTarget bit: %u\n", indent, "", bit);
            break;
        }
        case QkConditionType_ClReg: {
            printf("%*sTarget register:\n", indent, "");
            const QkClassicalRegister *creg = qk_control_flow_switch_target_register(cf_inst);
            inspect_register(creg, indent + 2);
            break;
        }
        case QkConditionType_Expr: {
            printf("%*sTarget expression:\n", indent, "");
            const QkExprNode *expr = qk_control_flow_switch_target_expr(cf_inst);
            inspect_expr(expr, indent + 2);
            break;
        }
        }

        // Inspect the Switch instruction cases
        size_t num_cases = qk_control_flow_switch_num_cases(cf_inst);
        printf("%*sNumber of cases: %zu\n", indent, "", num_cases);

        for (size_t case_idx = 0; case_idx < num_cases; case_idx++) {
            printf("%*sCase %zu:\n", indent, "", case_idx);

            uint64_t bit_width = qk_control_flow_switch_case_labels_bit_width(cf_inst, case_idx);

            if (bit_width <= 64) {
                QkSwitchCaseLabels labels = qk_control_flow_switch_case_labels_uint(cf_inst, case_idx);
                printf("%*sLabels (%zu): [", indent + 2, "", labels.num_labels);
                for (size_t label = 0; label < labels.num_labels; label++) {
                    printf("%lu%s", labels.labels[label], label < labels.num_labels - 1 ? ", " : "");
                }
                printf("]\n");

                if (labels.num_labels > 0) {
                    qk_control_flow_switch_case_labels_clear(&labels);
                }
            } else {
                printf("%*sLabel width (%lu bits) too large for direct display\n", indent + 2, "",
                    bit_width);
            }

            if (qk_control_flow_switch_is_case_default(cf_inst, case_idx)) {
                printf("%*sThis is the DEFAULT case\n", indent + 2, "");
            }
        }
    }

    void inspect_control_flow_instruction(const QkControlFlowInstruction *cf_inst, int indent) {
        QkControlFlowKind cf_type = qk_control_flow_kind(cf_inst);
        printf("%*s=== Control Flow: kind - %d ===\n", indent, "", cf_type);

        switch (cf_type) {
        case QkControlFlowKind_Box:
            inspect_box(cf_inst, indent + 2);
            break;
        case QkControlFlowKind_BreakLoop:
            printf("%*sBreak loop instruction\n", indent + 2, "");
            break;
        case QkControlFlowKind_ContinueLoop:
            printf("%*sContinue loop instruction\n", indent + 2, "");
            break;
        case QkControlFlowKind_ForLoop:
            inspect_for_loop(cf_inst, indent + 2);
            break;
        case QkControlFlowKind_IfElse:
            printf("%*sInspecting IfElse instruction\n", indent + 2, "");
            inspect_condition(cf_inst, indent + 2);
            break;
        case QkControlFlowKind_While:
            printf("%*sInspecting While instruction\n", indent + 2, "");
            inspect_condition(cf_inst, indent + 2);
            break;
        case QkControlFlowKind_Switch:
            inspect_switch(cf_inst, indent + 2);
            break;
        }

        size_t num_blocks = qk_control_flow_num_blocks(cf_inst);
        printf("%*sNumber of blocks: %zu\n", indent, "", num_blocks);

        for (size_t block = 0; block < num_blocks; block++) {
            printf("%*s--- Block %zu ---\n", indent, "", block);
            const QkCircuit *block_circuit = qk_control_flow_block_circuit(cf_inst, block);

            // Go deeper in the hierarchy
            inspect_circuit(block_circuit, cf_inst, indent + 2);
        }
    }

    void inspect_circuit(const QkCircuit *circuit, const QkControlFlowInstruction *parent_cf,
                        int indent) {
        size_t num_instructions = qk_circuit_num_instructions(circuit);
        printf("%*sCircuit has %zu instructions\n", indent, "", num_instructions);

        for (size_t inst_idx = 0; inst_idx < num_instructions; inst_idx++) {
            QkCircuitInstruction inst;
            qk_circuit_get_instruction(circuit, inst_idx, &inst);

            QkOperationKind kind = qk_circuit_instruction_kind(circuit, inst_idx);

            if (kind == QkOperationKind_ControlFlow) {
                QkControlFlowInstruction *cf_inst =
                    qk_circuit_get_control_flow_instruction(circuit, inst_idx, parent_cf);

                inspect_control_flow_instruction(cf_inst, indent);

                qk_control_flow_instruction_free(cf_inst);
            } else {
                printf("%*sInstruction %zu: Standard gate/operation\n", indent, "", inst_idx);

                // Inspect qubit mapping, if one exists
                const uint32_t *qubit_mapping = parent_cf ? qk_control_flow_qubit_map(parent_cf) : NULL;
                if (inst.num_qubits > 0) {
                    printf("%*s  Qubits: [", indent, "");
                    for (uint32_t qubit = 0; qubit < inst.num_qubits; qubit++) {
                        uint32_t mapped_qubit =
                            qubit_mapping ? qubit_mapping[inst.qubits[qubit]] : inst.qubits[qubit];
                        printf("%u%s", mapped_qubit, qubit < inst.num_qubits - 1 ? ", " : "");
                    }
                    printf("]\n");
                }

                // Inspect clbit mapping, if one exists
                const uint32_t *clbit_mapping = parent_cf ? qk_control_flow_clbit_map(parent_cf) : NULL;
                if (inst.num_clbits > 0) {
                    printf("%*s  Clbits: [", indent, "");
                    for (uint32_t clbit = 0; clbit < inst.num_clbits; clbit++) {
                        uint32_t mapped_clbit =
                            clbit_mapping ? clbit_mapping[inst.clbits[clbit]] : inst.clbits[clbit];
                        printf("%u%s", mapped_clbit, clbit < inst.num_clbits - 1 ? ", " : "");
                    }
                    printf("]\n");
                }
            }

            qk_circuit_instruction_clear(&inst);
        }
    }

Refer to the :doc:`qk-control-flow` and :doc:`qk-classical-expressions` documentation pages for 
more information about the C API functions and types.