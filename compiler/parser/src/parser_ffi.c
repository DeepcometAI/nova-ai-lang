/*
 * parser_ffi.c — shared-library friendly wrapper around the C lexer+parser.
 */

#include "../include/parser_ffi.h"
#include "../include/parser.h"
#include "../include/ast.h"
#include "../../lexer/include/lexer.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} StrBuf;

static void sb_init(StrBuf *sb) {
    sb->data = NULL;
    sb->len = 0;
    sb->cap = 0;
}

static void sb_free(StrBuf *sb) {
    free(sb->data);
    sb->data = NULL;
    sb->len = 0;
    sb->cap = 0;
}

static int sb_reserve(StrBuf *sb, size_t extra) {
    size_t need = sb->len + extra + 1; /* +1 for NUL */
    if (need <= sb->cap) return 1;
    size_t new_cap = (sb->cap == 0) ? 256 : sb->cap;
    while (new_cap < need) new_cap *= 2;
    char *tmp = (char *)realloc(sb->data, new_cap);
    if (!tmp) return 0;
    sb->data = tmp;
    sb->cap = new_cap;
    return 1;
}

static int sb_append_bytes(StrBuf *sb, const char *s, size_t n) {
    if (!sb_reserve(sb, n)) return 0;
    memcpy(sb->data + sb->len, s, n);
    sb->len += n;
    sb->data[sb->len] = '\0';
    return 1;
}

static int sb_append_cstr(StrBuf *sb, const char *s) {
    return sb_append_bytes(sb, s, strlen(s));
}

static int sb_append_indent(StrBuf *sb, int indent) {
    for (int i = 0; i < indent; i++) {
        if (!sb_append_cstr(sb, "  ")) return 0;
    }
    return 1;
}

static int sb_append_novstr(StrBuf *sb, NovString s) {
    if (!s.ptr || s.len == 0) return 1;
    return sb_append_bytes(sb, s.ptr, s.len);
}

static int sb_append_u32(StrBuf *sb, unsigned v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%u", v);
    return sb_append_cstr(sb, buf);
}

static int sb_append_i64(StrBuf *sb, long long v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%lld", v);
    return sb_append_cstr(sb, buf);
}

static int sb_append_f64(StrBuf *sb, double v) {
    char buf[64];
    /* %g keeps it compact */
    snprintf(buf, sizeof(buf), "%g", v);
    return sb_append_cstr(sb, buf);
}

static int dump_node(StrBuf *sb, const NovAstNode *node, int indent) {
    if (!node) {
        return sb_append_indent(sb, indent) && sb_append_cstr(sb, "<null>\n");
    }

    if (!sb_append_indent(sb, indent)) return 0;
    if (!sb_append_cstr(sb, "[")) return 0;
    if (!sb_append_u32(sb, node->line)) return 0;
    if (!sb_append_cstr(sb, ":")) return 0;
    if (!sb_append_u32(sb, node->col)) return 0;
    if (!sb_append_cstr(sb, "] ")) return 0;
    if (!sb_append_cstr(sb, nova_ast_node_type_name(node->type))) return 0;

    switch (node->type) {
        case AST_IDENT:
            if (!sb_append_cstr(sb, " '")) return 0;
            if (!sb_append_novstr(sb, node->as.ident.name)) return 0;
            if (!sb_append_cstr(sb, "'")) return 0;
            break;
        case AST_INT_LIT:
            if (!sb_append_cstr(sb, " ")) return 0;
            if (!sb_append_i64(sb, (long long)node->as.int_lit.value)) return 0;
            break;
        case AST_FLOAT_LIT:
            if (!sb_append_cstr(sb, " ")) return 0;
            if (!sb_append_f64(sb, node->as.float_lit.value)) return 0;
            break;
        case AST_UNIT_LIT:
            if (!sb_append_cstr(sb, " ")) return 0;
            if (!sb_append_f64(sb, node->as.unit_lit.value)) return 0;
            if (!sb_append_cstr(sb, "[")) return 0;
            if (!sb_append_novstr(sb, node->as.unit_lit.unit_str)) return 0;
            if (!sb_append_cstr(sb, "]")) return 0;
            break;
        case AST_STRING_LIT:
            if (!sb_append_cstr(sb, " \"")) return 0;
            if (!sb_append_novstr(sb, node->as.string_lit.value)) return 0;
            if (!sb_append_cstr(sb, "\"")) return 0;
            break;
        case AST_BOOL_LIT:
            if (!sb_append_cstr(sb, " ")) return 0;
            if (!sb_append_cstr(sb, node->as.bool_lit.value ? "true" : "false")) return 0;
            break;
        case AST_MISSION_DECL:
            if (!sb_append_cstr(sb, " '")) return 0;
            if (!sb_append_novstr(sb, node->as.mission_decl.name)) return 0;
            if (!sb_append_cstr(sb, "'")) return 0;
            if (node->as.mission_decl.is_parallel) {
                if (!sb_append_cstr(sb, " [parallel]")) return 0;
            }
            break;
        case AST_BINARY_EXPR:
            if (!sb_append_cstr(sb, " op=")) return 0;
            if (!sb_append_cstr(sb, nova_token_type_name(node->as.binary_expr.op))) return 0;
            break;
        case AST_UNARY_EXPR:
            if (!sb_append_cstr(sb, " op=")) return 0;
            if (!sb_append_cstr(sb, nova_token_type_name(node->as.unary_expr.op))) return 0;
            break;
        case AST_NAMED_TYPE:
            if (!sb_append_cstr(sb, " '")) return 0;
            if (!sb_append_novstr(sb, node->as.named_type.name)) return 0;
            if (!sb_append_cstr(sb, "'")) return 0;
            break;
        default:
            break;
    }

    if (!sb_append_cstr(sb, "\n")) return 0;

    /* Recurse into children (mirror ast.c printer for core nodes) */
    switch (node->type) {
        case AST_PROGRAM:
            for (size_t i = 0; i < node->as.program.items.count; i++) {
                if (!dump_node(sb, node->as.program.items.items[i], indent + 1)) return 0;
            }
            break;
        case AST_MISSION_DECL:
            for (size_t i = 0; i < node->as.mission_decl.params.count; i++) {
                if (!dump_node(sb, node->as.mission_decl.params.items[i], indent + 1)) return 0;
            }
            if (!dump_node(sb, node->as.mission_decl.ret_type, indent + 1)) return 0;
            if (!dump_node(sb, node->as.mission_decl.body, indent + 1)) return 0;
            break;
        case AST_BLOCK:
            for (size_t i = 0; i < node->as.block.stmts.count; i++) {
                if (!dump_node(sb, node->as.block.stmts.items[i], indent + 1)) return 0;
            }
            break;
        case AST_EXPR_STMT:
            if (!dump_node(sb, node->as.expr_stmt.expr, indent + 1)) return 0;
            break;
        case AST_CALL_EXPR:
            if (!dump_node(sb, node->as.call_expr.callee, indent + 1)) return 0;
            for (size_t i = 0; i < node->as.call_expr.args.count; i++) {
                if (!dump_node(sb, node->as.call_expr.args.items[i], indent + 1)) return 0;
            }
            break;
        case AST_TRANSMIT_EXPR:
            if (!dump_node(sb, node->as.transmit_expr.arg, indent + 1)) return 0;
            break;
        case AST_RETURN_STMT:
            if (!dump_node(sb, node->as.return_stmt.value, indent + 1)) return 0;
            break;
        case AST_BINARY_EXPR:
            if (!dump_node(sb, node->as.binary_expr.lhs, indent + 1)) return 0;
            if (!dump_node(sb, node->as.binary_expr.rhs, indent + 1)) return 0;
            break;
        case AST_UNARY_EXPR:
            if (!dump_node(sb, node->as.unary_expr.operand, indent + 1)) return 0;
            break;
        case AST_LET_DECL:
            if (!dump_node(sb, node->as.let_decl.type_node, indent + 1)) return 0;
            if (!dump_node(sb, node->as.let_decl.init, indent + 1)) return 0;
            break;
        case AST_IF_STMT:
            if (!dump_node(sb, node->as.if_stmt.cond, indent + 1)) return 0;
            if (!dump_node(sb, node->as.if_stmt.then_block, indent + 1)) return 0;
            if (!dump_node(sb, node->as.if_stmt.else_block, indent + 1)) return 0;
            break;
        case AST_FOR_STMT:
            if (!dump_node(sb, node->as.for_stmt.iter, indent + 1)) return 0;
            if (!dump_node(sb, node->as.for_stmt.body, indent + 1)) return 0;
            break;
        case AST_WHILE_STMT:
            if (!dump_node(sb, node->as.while_stmt.cond, indent + 1)) return 0;
            if (!dump_node(sb, node->as.while_stmt.body, indent + 1)) return 0;
            break;
        default:
            break;
    }

    return 1;
}

static char *sb_take(StrBuf *sb) {
    if (!sb->data) {
        char *s = (char *)malloc(1);
        if (s) s[0] = '\0';
        return s;
    }
    char *out = sb->data;
    sb->data = NULL;
    sb->len = 0;
    sb->cap = 0;
    return out;
}

static char *dup_cstr(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n + 1);
    return out;
}

static char *build_errors(const NovaLexer *lex, const NovParser *p) {
    StrBuf sb;
    sb_init(&sb);

    if (nova_lexer_had_errors(lex)) {
        sb_append_cstr(&sb, "lexer errors:\n");
        for (int i = 0; i < lex->error_count; i++) {
            char linebuf[256];
            snprintf(linebuf, sizeof(linebuf), "  %u:%u: %s\n",
                     lex->errors[i].line, lex->errors[i].col, lex->errors[i].message);
            sb_append_cstr(&sb, linebuf);
        }
    }

    if (nova_parser_had_errors(p)) {
        sb_append_cstr(&sb, "parser errors:\n");
        for (int i = 0; i < p->error_count; i++) {
            char linebuf[512];
            snprintf(linebuf, sizeof(linebuf), "  %u:%u: %s\n",
                     p->errors[i].line, p->errors[i].col, p->errors[i].message);
            sb_append_cstr(&sb, linebuf);
        }
    }

    return sb_take(&sb);
}

int nova_parse_dump(const char *src, size_t src_len, const char *filename,
                    char **out_dump, char **out_errors) {
    if (!out_dump || !out_errors) return 2;
    *out_dump = NULL;
    *out_errors = NULL;

    if (!src) {
        *out_dump = dup_cstr("");
        *out_errors = dup_cstr("error: src is NULL\n");
        return 2;
    }

    NovaLexer lex;
    nova_lexer_init(&lex, src, src_len, filename);

    NovParser p;
    nova_parser_init(&p, &lex);

    NovAstNode *root = nova_parse(&p);

    StrBuf dump;
    sb_init(&dump);
    dump_node(&dump, root, 0);
    *out_dump = sb_take(&dump);

    *out_errors = build_errors(&lex, &p);

    int had_err = nova_lexer_had_errors(&lex) || nova_parser_had_errors(&p);

    nova_ast_free(root);
    nova_lexer_free(&lex);

    return had_err ? 1 : 0;
}

void nova_ffi_free(char *ptr) {
    free(ptr);
}

