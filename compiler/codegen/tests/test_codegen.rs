//! NOVA CodeGen — integration tests
//!
//! These tests exercise the IR emitter, FFI codegen, and parallel scheduler
//! without requiring a full LLVM installation. They validate the data
//! structures, type mappings, and code-generation logic in isolation.

use nova_codegen::{
    IREmitter, CodegenError, ModuleBuilder, FunctionBuilder, FFICodegen, ParallelScheduler,
};
use nova_codegen::ir_emitter::NovaType;

// ── IREmitter tests ───────────────────────────────────────────────────────────

#[test]
fn emitter_new() {
    let emitter = IREmitter::new("test_module");
    assert!(!emitter.module_name().is_empty());
}

#[test]
fn emitter_module_name() {
    let emitter = IREmitter::new("hello_world");
    assert_eq!(emitter.module_name(), "hello_world");
}

#[test]
fn nova_type_float_display() {
    let t = NovaType::Float;
    assert_eq!(format!("{}", t), "double");
}

#[test]
fn nova_type_int_display() {
    let t = NovaType::Int;
    assert_eq!(format!("{}", t), "i64");
}

#[test]
fn nova_type_bool_display() {
    let t = NovaType::Bool;
    assert_eq!(format!("{}", t), "i1");
}

#[test]
fn nova_type_void_display() {
    let t = NovaType::Void;
    assert_eq!(format!("{}", t), "void");
}

#[test]
fn codegen_error_display_type_error() {
    let e = CodegenError::TypeError("Float expected".into());
    let s = format!("{}", e);
    assert!(s.contains("Float expected"));
}

#[test]
fn codegen_error_display_unknown_function() {
    let e = CodegenError::UnknownFunction("delta_v".into());
    let s = format!("{}", e);
    assert!(s.contains("delta_v"));
}

#[test]
fn codegen_error_display_invalid_operand() {
    let e = CodegenError::InvalidOperand("left hand side".into());
    let s = format!("{}", e);
    assert!(s.contains("left hand side"));
}

#[test]
fn codegen_error_display_ir_failed() {
    let e = CodegenError::IRGenerationFailed("out of memory".into());
    let s = format!("{}", e);
    assert!(s.contains("out of memory"));
}

// ── ModuleBuilder tests ───────────────────────────────────────────────────────

#[test]
fn module_builder_new() {
    let mb = ModuleBuilder::new("nova_test");
    assert_eq!(mb.name(), "nova_test");
}

#[test]
fn module_builder_declare_function() {
    let mut mb = ModuleBuilder::new("nova_test");
    mb.declare_function("main", NovaType::Void, vec![]);
    assert!(mb.has_function("main"));
}

#[test]
fn module_builder_no_function_initially() {
    let mb = ModuleBuilder::new("empty");
    assert!(!mb.has_function("delta_v"));
}

#[test]
fn module_builder_declare_multiple_functions() {
    let mut mb = ModuleBuilder::new("multi");
    mb.declare_function("f1", NovaType::Float, vec![NovaType::Float]);
    mb.declare_function("f2", NovaType::Int,   vec![NovaType::Int, NovaType::Int]);
    mb.declare_function("f3", NovaType::Void,  vec![]);
    assert!(mb.has_function("f1"));
    assert!(mb.has_function("f2"));
    assert!(mb.has_function("f3"));
    assert!(!mb.has_function("f4"));
}

#[test]
fn module_builder_function_count() {
    let mut mb = ModuleBuilder::new("count_test");
    assert_eq!(mb.function_count(), 0);
    mb.declare_function("a", NovaType::Void, vec![]);
    mb.declare_function("b", NovaType::Float, vec![]);
    assert_eq!(mb.function_count(), 2);
}

// ── FunctionBuilder tests ─────────────────────────────────────────────────────

#[test]
fn function_builder_new() {
    let fb = FunctionBuilder::new("delta_v", NovaType::Float);
    assert_eq!(fb.name(), "delta_v");
    assert_eq!(fb.return_type(), NovaType::Float);
}

#[test]
fn function_builder_add_param() {
    let mut fb = FunctionBuilder::new("add", NovaType::Float);
    fb.add_param("a", NovaType::Float);
    fb.add_param("b", NovaType::Float);
    assert_eq!(fb.param_count(), 2);
}

#[test]
fn function_builder_emit_ret() {
    let mut fb = FunctionBuilder::new("identity", NovaType::Float);
    fb.add_param("x", NovaType::Float);
    let result = fb.emit_return("x");
    assert!(result.is_ok());
}

#[test]
fn function_builder_emit_add() {
    let mut fb = FunctionBuilder::new("add", NovaType::Float);
    fb.add_param("a", NovaType::Float);
    fb.add_param("b", NovaType::Float);
    let result = fb.emit_add("a", "b");
    assert!(result.is_ok());
}

#[test]
fn function_builder_emit_mul() {
    let mut fb = FunctionBuilder::new("mul", NovaType::Float);
    fb.add_param("a", NovaType::Float);
    fb.add_param("b", NovaType::Float);
    let result = fb.emit_mul("a", "b");
    assert!(result.is_ok());
}

#[test]
fn function_builder_instructions_accumulate() {
    let mut fb = FunctionBuilder::new("chain", NovaType::Float);
    fb.add_param("x", NovaType::Float);
    fb.emit_add("x", "x").unwrap();
    fb.emit_mul("x", "x").unwrap();
    assert!(fb.instruction_count() >= 2);
}

// ── FFICodegen tests ──────────────────────────────────────────────────────────

#[test]
fn ffi_codegen_new() {
    let ffi = FFICodegen::new();
    assert_eq!(ffi.extern_count(), 0);
}

#[test]
fn ffi_codegen_declare_extern_c() {
    let mut ffi = FFICodegen::new();
    ffi.declare_extern("malloc", NovaType::Void, vec![NovaType::Int]);
    assert!(ffi.has_extern("malloc"));
}

#[test]
fn ffi_codegen_declare_multiple() {
    let mut ffi = FFICodegen::new();
    ffi.declare_extern("free",   NovaType::Void,  vec![NovaType::Int]);
    ffi.declare_extern("printf", NovaType::Int,   vec![]);
    ffi.declare_extern("sqrt",   NovaType::Float, vec![NovaType::Float]);
    assert_eq!(ffi.extern_count(), 3);
}

#[test]
fn ffi_codegen_emit_declaration() {
    let mut ffi = FFICodegen::new();
    ffi.declare_extern("sqrt", NovaType::Float, vec![NovaType::Float]);
    let decl = ffi.emit_declaration("sqrt").unwrap();
    assert!(decl.contains("sqrt"));
}

#[test]
fn ffi_codegen_unknown_extern_errors() {
    let ffi = FFICodegen::new();
    let result = ffi.emit_declaration("nonexistent");
    assert!(result.is_err());
}

// ── ParallelScheduler tests ───────────────────────────────────────────────────

#[test]
fn scheduler_new() {
    let sched = ParallelScheduler::new();
    assert_eq!(sched.task_count(), 0);
}

#[test]
fn scheduler_add_task() {
    let mut sched = ParallelScheduler::new();
    sched.add_task("reduce_catalogue");
    assert_eq!(sched.task_count(), 1);
}

#[test]
fn scheduler_add_multiple_tasks() {
    let mut sched = ParallelScheduler::new();
    sched.add_task("filter");
    sched.add_task("map");
    sched.add_task("reduce");
    assert_eq!(sched.task_count(), 3);
}

#[test]
fn scheduler_emit_parallel_preamble() {
    let sched = ParallelScheduler::new();
    let preamble = sched.emit_preamble();
    // Should contain threading or parallel marker
    assert!(!preamble.is_empty());
}

#[test]
fn scheduler_strategy_data() {
    use nova_codegen::ParallelScheduler;
    let sched = ParallelScheduler::with_strategy(nova_codegen::parallel_scheduler::ParallelStrategy::Data);
    let preamble = sched.emit_preamble();
    assert!(preamble.to_lowercase().contains("data") || !preamble.is_empty());
}

#[test]
fn scheduler_strategy_task() {
    use nova_codegen::ParallelScheduler;
    let sched = ParallelScheduler::with_strategy(nova_codegen::parallel_scheduler::ParallelStrategy::Task);
    let preamble = sched.emit_preamble();
    assert!(!preamble.is_empty());
}

// ── End-to-end minimal codegen scenario ──────────────────────────────────────

#[test]
fn e2e_hello_universe_codegen() {
    // Simulate: mission main() -> Void { transmit("Hello, universe!") }
    let mut mb = ModuleBuilder::new("hello_universe");
    mb.declare_function("main", NovaType::Void, vec![]);
    assert!(mb.has_function("main"));
    // The module should be non-empty once functions are declared
    let ir = mb.emit_module_ir();
    assert!(!ir.is_empty());
}

#[test]
fn e2e_delta_v_function_signature() {
    // mission delta_v(isp: Float, m_wet: Float, m_dry: Float) -> Float
    let mut mb = ModuleBuilder::new("delta_v_module");
    mb.declare_function(
        "delta_v",
        NovaType::Float,
        vec![NovaType::Float, NovaType::Float, NovaType::Float],
    );
    assert!(mb.has_function("delta_v"));
    assert_eq!(mb.function_count(), 1);
}

#[test]
fn e2e_parallel_mission_codegen() {
    // parallel mission process(data: Array) -> Array
    let mut mb  = ModuleBuilder::new("parallel_test");
    let mut sched = ParallelScheduler::new();
    mb.declare_function("process", NovaType::Void, vec![]);
    sched.add_task("process");
    assert!(mb.has_function("process"));
    assert_eq!(sched.task_count(), 1);
}

#[test]
fn e2e_ffi_c_malloc_free() {
    let mut ffi = FFICodegen::new();
    ffi.declare_extern("malloc", NovaType::Void, vec![NovaType::Int]);
    ffi.declare_extern("free",   NovaType::Void, vec![NovaType::Int]);
    let malloc_decl = ffi.emit_declaration("malloc").unwrap();
    let free_decl   = ffi.emit_declaration("free").unwrap();
    assert!(malloc_decl.contains("malloc"));
    assert!(free_decl.contains("free"));
}
