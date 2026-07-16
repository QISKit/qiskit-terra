// This code is part of Qiskit.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include "common.h"
#include <complex.h>
#include <inttypes.h>
#include <math.h>
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// TODO: remove these forward declarations
// These are used for generating classical expressions for testing. They are non-public C API
// functions which should be removed once we have C API for generating classical expressions.
QkExprNode *inner_test_expression_structs();
QkExprNode *inner_test_unary_expr_ops(QkUnaryOpType);
QkExprNode *inner_test_binary_expr_ops(QkBinaryOpType);
QkExprNode *inner_test_expr_kinds_and_types(QkExprNodeKind, QkExprTypeInfo);
QkExprNode *inner_test_value(QkExprType, bool, QkDurationInfo, double, uint64_t);
void inned_test_old_style_vars(QkExprNode **);
void *inner_expr_free(QkExprNode *);

/*
 * Test that expression tree structure is captured correctly via the information structs
 */
static int test_expr_info_structs(void) {
    // Build an expression with the following structure:
    // Index(
    //     'v1',
    //     Cast(
    //         Unary(LOGIC_NOT,
    //             Binary(GREATER,
    //                 'v1',
    //                 Value(5)
    //             )
    //         )
    //         Uint(1)
    //     )
    // )
    QkExprNode *expr = inner_test_expression_structs();

    int result = Ok;

    QkExprNodeKind kind = qk_expr_kind(expr);
    if (kind != QkExprNodeKind_Index) {
        printf("Expected Index node, got %d\n", kind);
        result = EqualityError;
        goto cleanup;
    }

    QkIndexExprInfo index_info = qk_expr_index_info(expr);
    QkExprNodeKind target_kind = qk_expr_kind(index_info.target);
    if (target_kind != QkExprNodeKind_Var) {
        printf("Expected Var node for target, got %d\n", target_kind);
        result = EqualityError;
        goto cleanup;
    }

    QkExprNodeKind index_kind = qk_expr_kind(index_info.index);
    if (index_kind != QkExprNodeKind_Cast) {
        printf("Expected Cast node for index, got %d\n", index_kind);
        result = EqualityError;
        goto cleanup;
    }

    QkCastExprInfo cast_info = qk_expr_cast_info(index_info.index);
    if (cast_info.ty.ty != QkExprType_Uint || cast_info.ty.width != 1) {
        printf("Expected Cast to Uint(1), got type %d width %d\n", cast_info.ty.ty,
               cast_info.ty.width);
        result = EqualityError;
        goto cleanup;
    }

    QkExprNodeKind cast_operand_kind = qk_expr_kind(cast_info.operand);
    if (cast_operand_kind != QkExprNodeKind_Unary) {
        printf("Expected Unary node for cast operand, got %d\n", cast_operand_kind);
        result = EqualityError;
        goto cleanup;
    }

    QkUnaryExprInfo unary_info = qk_expr_unary_info(cast_info.operand);
    if (unary_info.op != QkUnaryOpType_LogicNot) {
        printf("Expected LogicNot operation, got %d\n", unary_info.op);
        result = EqualityError;
        goto cleanup;
    }

    QkExprNodeKind unary_operand_kind = qk_expr_kind(unary_info.operand);
    if (unary_operand_kind != QkExprNodeKind_Binary) {
        printf("Expected Binary node for unary operand, got %d\n", unary_operand_kind);
        result = EqualityError;
        goto cleanup;
    }

    QkBinaryExprInfo binary_info = qk_expr_binary_info(unary_info.operand);
    if (binary_info.op != QkBinaryOpType_Greater) {
        printf("Expected Greater operation, got %d\n", binary_info.op);
        result = EqualityError;
        goto cleanup;
    }

    QkExprNodeKind left_kind = qk_expr_kind(binary_info.left);
    if (left_kind != QkExprNodeKind_Var) {
        printf("Expected Var node for left operand, got %d\n", left_kind);
        result = EqualityError;
        goto cleanup;
    }

    QkExprNodeKind right_kind = qk_expr_kind(binary_info.right);
    if (right_kind != QkExprNodeKind_Value) {
        printf("Expected Value node for right operand, got %d\n", right_kind);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    inner_expr_free(expr);

    return result;
}

/*
 * Test that unary and binary op types are assigned and retrieved correctly
 */
static int test_op_types_roundtrip(void) {
    int result = Ok;
    QkExprNode *expr = NULL;

    for (int op = QkUnaryOpType_BitNot; op <= QkUnaryOpType_Negate; op++) {
        QkUnaryOpType expected_op = (QkUnaryOpType)op;

        expr = inner_test_unary_expr_ops(expected_op);

        QkExprNodeKind kind = qk_expr_kind(expr);
        if (kind != QkExprNodeKind_Unary) {
            printf("Expected Unary expression node, got %d\n", kind);
            result = EqualityError;
            goto cleanup;
        }

        QkUnaryExprInfo unary_info = qk_expr_unary_info(expr);
        if (unary_info.op != expected_op) {
            printf("Unary op type mismatch: expected %d, got %d\n", expected_op, unary_info.op);
            result = EqualityError;
            goto cleanup;
        }

        inner_expr_free(expr);
    }

    for (size_t op = QkBinaryOpType_BitAnd; op <= QkBinaryOpType_Div; op++) {
        QkBinaryOpType expected_op = (QkBinaryOpType)op;

        expr = inner_test_binary_expr_ops(expected_op);

        QkExprNodeKind kind = qk_expr_kind(expr);
        if (kind != QkExprNodeKind_Binary) {
            printf("Expected Binary expression node, got %d\n", kind);
            result = EqualityError;
            goto cleanup;
        }

        QkBinaryExprInfo binary_info = qk_expr_binary_info(expr);
        if (binary_info.op != expected_op) {
            printf("Op type mismatch: expected %d, got %d\n", expected_op, binary_info.op);
            result = EqualityError;
            goto cleanup;
        }

        inner_expr_free(expr);
    }

    expr = NULL;

cleanup:
    if (expr)
        inner_expr_free(expr);

    return result;
}

/*
 * Test that kind and expr types round trip
 */
static int test_expr_kind_and_type(void) {
    int result = Ok;
    QkExprNodeKind kinds_to_check[] = {QkExprNodeKind_Unary, QkExprNodeKind_Binary,
                                       QkExprNodeKind_Cast, QkExprNodeKind_Index};
    QkExprNode *expr = NULL;

    for (size_t kind_idx = 0; kind_idx < 4; kind_idx++) {
        QkExprNodeKind kind = kinds_to_check[kind_idx];

        for (size_t ty = QkExprType_Bool; ty <= QkExprType_Uint; ty++) {
            QkExprTypeInfo type_info = {(QkExprType)ty, ty == QkExprType_Uint ? 8 : 0};

            expr = inner_test_expr_kinds_and_types(kind, type_info);

            QkExprNodeKind result_kind = qk_expr_kind(expr);
            if (result_kind != kind) {
                printf("Kind mismatch: expected %d, got %d\n", kind, result_kind);
                result = EqualityError;
                goto cleanup;
            }

            QkExprTypeInfo result_type;
            switch (result_kind) {
            case QkExprNodeKind_Unary: {
                QkUnaryExprInfo info = qk_expr_unary_info(expr);
                result_type = info.ty;
                break;
            }
            case QkExprNodeKind_Binary: {
                QkBinaryExprInfo info = qk_expr_binary_info(expr);
                result_type = info.ty;
                break;
            }
            case QkExprNodeKind_Cast: {
                QkCastExprInfo info = qk_expr_cast_info(expr);
                result_type = info.ty;
                break;
            }
            case QkExprNodeKind_Index: {
                QkIndexExprInfo info = qk_expr_index_info(expr);
                result_type = info.ty;
                break;
            }
            default:
                printf("Unexpected kind for type checking: %d\n", result_kind);
                result = EqualityError;
                goto cleanup;
            }

            if (result_type.ty != type_info.ty) {
                printf("Type mismatch: expected %d, got %d\n", type_info.ty, result_type.ty);
                result = EqualityError;
                goto cleanup;
            }

            if (type_info.ty == QkExprType_Uint && result_type.width != type_info.width) {
                printf("Width mismatch for Uint: expected %d, got %d\n", type_info.width,
                       result_type.width);
                result = EqualityError;
                goto cleanup;
            }

            inner_expr_free(expr);
        }
    }

    expr = NULL;

cleanup:
    if (expr)
        inner_expr_free(expr);

    return result;
}

/*
 * Test variable queries and property handling
 */
static int test_expr_var(void) {
    int result = Ok;
    QkExprNode *expr = NULL;

    for (uint8_t ty = QkExprType_Bool; ty <= QkExprType_Uint; ty++) {
        QkExprTypeInfo type_info = {ty, ty == QkExprType_Uint ? 8 : 0};

        expr = inner_test_expr_kinds_and_types(QkExprNodeKind_Var, type_info);
        QkExprNodeKind kind = qk_expr_kind(expr);
        if (kind != QkExprNodeKind_Var) {
            printf("Expected Var node, got %d\n", kind);
            result = EqualityError;
            goto cleanup;
        }

        const QkVar *var = qk_expr_as_var(expr);
        if (var == NULL) {
            printf("qk_expr_as_var returned NULL for Var expression\n");
            result = EqualityError;
            goto cleanup;
        }

        char *name = qk_var_name(var);
        if (name == NULL) {
            printf("qk_var_name returned NULL\n");
            result = EqualityError;
            goto cleanup;
        }

        if (strcmp(name, "test_var") != 0) {
            printf("Expected var name 'test_var', got '%s'\n", name);
            qk_str_free(name);
            result = EqualityError;
            goto cleanup;
        }
        qk_str_free(name);

        QkExprTypeInfo var_type = qk_var_type_info(var);
        if (var_type.ty != type_info.ty) {
            printf("Var type mismatch: expected %d, got %d\n", type_info.ty, var_type.ty);
            result = EqualityError;
            goto cleanup;
        }

        if (type_info.ty == QkExprType_Uint && var_type.width != type_info.width) {
            printf("Var width mismatch: expected %d, got %d\n", type_info.width, var_type.width);
            result = EqualityError;
            goto cleanup;
        }

        inner_expr_free(expr);
    }

    expr = NULL;

    QkExprNode *var_nodes[2];
    inned_test_old_style_vars(var_nodes);

    // The first var is a Bit variable
    const QkVar *var = qk_expr_as_var(var_nodes[0]);
    QkExprTypeInfo var_type_info = qk_var_type_info(var);

    if (var_type_info.ty != QkExprType_Bool) {
        printf("Expected var type to be QkExprType_Bool, got %d\n", var_type_info.ty);
        result = EqualityError;
        goto cleanup_vars;
    }

    char *name = qk_var_name(var);
    if (name != NULL) {
        printf("Expected var name to be NULL, got %s\n", name);
        qk_str_free(name);
        result = EqualityError;
        goto cleanup_vars;
    }

    // The second var is a Register variable
    var = qk_expr_as_var(var_nodes[1]);
    var_type_info = qk_var_type_info(var);

    if (var_type_info.ty != QkExprType_Uint) {
        printf("Expected var type to be QkExprType_Uint, got %d\n", var_type_info.ty);
        result = EqualityError;
        goto cleanup_vars;
    }

    name = qk_var_name(var);
    if (name != NULL) {
        printf("Expected var name to be NULL, got %s\n", name);
        qk_str_free(name);
        result = EqualityError;
        goto cleanup_vars;
    }

cleanup_vars:
    inner_expr_free(var_nodes[0]);
    inner_expr_free(var_nodes[1]);

cleanup:
    if (expr)
        inner_expr_free(expr);

    return result;
}

/*
 * Test stretch queries and property handling
 */
static int test_expr_stretch(void) {
    int result = Ok;
    QkExprNode *expr = NULL;

    for (uint8_t ty = QkExprType_Bool; ty <= QkExprType_Uint; ty++) {
        QkExprTypeInfo type_info = {ty, ty == QkExprType_Uint ? 8 : 0};

        expr = inner_test_expr_kinds_and_types(QkExprNodeKind_Stretch, type_info);

        QkExprNodeKind kind = qk_expr_kind(expr);
        if (kind != QkExprNodeKind_Stretch) {
            printf("Expected Stretch node, got %d\n", kind);
            result = EqualityError;
            goto cleanup;
        }

        const QkStretch *stretch = qk_expr_as_stretch(expr);
        if (stretch == NULL) {
            printf("qk_expr_as_stretch returned NULL for Stretch expression\n");
            result = EqualityError;
            goto cleanup;
        }

        char *name = qk_stretch_name(stretch);
        if (name == NULL) {
            printf("qk_stretch_name returned NULL\n");
            result = EqualityError;
            goto cleanup;
        }

        if (strcmp(name, "test_stretch") != 0) {
            printf("Expected stretch name 'test_stretch', got '%s'\n", name);
            qk_str_free(name);
            result = EqualityError;
            goto cleanup;
        }
        qk_str_free(name);

        inner_expr_free(expr);
    }

    expr = NULL;

cleanup:
    if (expr)
        inner_expr_free(expr);

    return result;
}

/*
 * Test operations with values
 */
static int test_expr_value(void) {
    int result = Ok;
    QkDurationInfo duration_info = {QkDurationType_Dt, {.dt = 12345}};
    QkExprNode *expr = NULL;

    for (uint8_t ty = QkExprType_Bool; ty <= QkExprType_Uint; ty++) {
        expr = inner_test_value((QkExprType)ty, true, duration_info, 3.14, 12345);

        QkExprNodeKind kind = qk_expr_kind(expr);
        if (kind != QkExprNodeKind_Value) {
            printf("Expected Value node for Bool, got %d\n", kind);
            result = EqualityError;
            goto cleanup;
        }

        const QkValue *value = qk_expr_as_value(expr);
        if (value == NULL) {
            printf("qk_expr_as_value returned NULL for Bool value\n");
            result = EqualityError;
            goto cleanup;
        }

        QkExprTypeInfo value_type_info = qk_value_type_info(value);
        switch (ty) {
        case QkExprType_Bool:
            if (value_type_info.ty != QkExprType_Bool) {
                printf("Expected Bool type, got %d\n", value_type_info.ty);
                result = EqualityError;
                goto cleanup;
            }

            bool bool_val = qk_value_bool(value);
            if (bool_val != true) {
                printf("Expected true, got %d\n", bool_val);
                result = EqualityError;
                goto cleanup;
            }
            break;
        case QkExprType_Duration:
            if (value_type_info.ty != QkExprType_Duration) {
                printf("Expected Duration type, got %d\n", value_type_info.ty);
                result = EqualityError;
                goto cleanup;
            }

            QkDurationInfo result_duration = qk_value_duration_info(value);
            if (result_duration.ty != QkDurationType_Dt) {
                printf("Expected Dt duration type, got %d\n", result_duration.ty);
                result = EqualityError;
                goto cleanup;
            }

            if (result_duration.value.dt != 12345) {
                printf("Expected dt value 12345, got %" PRId64 "\n", result_duration.value.dt);
                result = EqualityError;
                goto cleanup;
            }
            break;
        case QkExprType_Float:
            if (value_type_info.ty != QkExprType_Float) {
                printf("Expected Float type, got %d\n", value_type_info.ty);
                result = EqualityError;
                goto cleanup;
            }

            double float_val = qk_value_float(value);
            if (float_val != 3.14) {
                printf("Expected float value 3.14159, got %f\n", float_val);
                result = EqualityError;
                goto cleanup;
            }
            break;
        case QkExprType_Uint:
            if (value_type_info.ty != QkExprType_Uint) {
                printf("Expected Uint type, got %d\n", value_type_info.ty);
                result = EqualityError;
                goto cleanup;
            }

            if (value_type_info.width != 4) {
                printf("Expected Uint width to be 4, got %" PRIu32 "\n", value_type_info.width);
                result = EqualityError;
                goto cleanup;
            }

            uint64_t uint_val = qk_value_uint(value);

            if (uint_val != 12345) {
                printf("Expected uint value 12345, got %" PRIu64 "\n", uint_val);
                result = EqualityError;
                goto cleanup;
            }
        }

        inner_expr_free(expr);
    }

    expr = NULL;

cleanup:
    if (expr)
        inner_expr_free(expr);

    return result;
}

int test_classical_expr(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_expr_info_structs);
    num_failed += RUN_TEST(test_op_types_roundtrip);
    num_failed += RUN_TEST(test_expr_kind_and_type);
    num_failed += RUN_TEST(test_expr_var);
    num_failed += RUN_TEST(test_expr_stretch);
    num_failed += RUN_TEST(test_expr_value);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
