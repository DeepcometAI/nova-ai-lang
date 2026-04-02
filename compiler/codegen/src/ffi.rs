//! Foreign Function Interface code generation
//!
//! Handles C FFI interop for NOVA missions calling C functions.

use crate::ir_emitter::{NovaType, CodegenError};

/// FFI binding for external C functions.
#[derive(Debug, Clone)]
pub struct ExternFunction {
    pub name: String,
    pub return_type: NovaType,
    pub parameters: Vec<NovaType>,
}

impl ExternFunction {
    pub fn new(name: String, return_type: NovaType) -> Self {
        ExternFunction {
            name,
            return_type,
            parameters: Vec::new(),
        }
    }

    pub fn add_parameter(&mut self, ty: NovaType) {
        self.parameters.push(ty);
    }

    pub fn emit_declaration(&self) -> String {
        let params = self
            .parameters
            .iter()
            .map(|ty| ty.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "declare {} @{}({})",
            self.return_type.to_string(),
            self.name,
            params
        )
    }
}

/// FFI code generator.
pub struct FFICodegen;

impl FFICodegen {
    pub fn new() -> Self {
        FFICodegen
    }

    pub fn generate_wrapper(
        &self,
        extern_func: &ExternFunction,
    ) -> Result<String, CodegenError> {
        let wrapper_name = format!("{}_wrapper", extern_func.name);
        let ret_ty = extern_func.return_type.to_string();

        // Name wrapper parameters deterministically: %arg0, %arg1, ...
        let param_list = extern_func
            .parameters
            .iter()
            .enumerate()
            .map(|(i, ty)| format!("{} %arg{}", ty.to_string(), i))
            .collect::<Vec<_>>()
            .join(", ");

        let arg_list = (0..extern_func.parameters.len())
            .map(|i| format!("%arg{}", i))
            .collect::<Vec<_>>()
            .join(", ");

        let mut wrapper = String::new();
        wrapper.push_str(&format!(
            "define {} @{}({}) {{\nentry:\n",
            ret_ty, wrapper_name, param_list
        ));

        match extern_func.return_type {
            NovaType::Void => {
                wrapper.push_str(&format!(
                    "  call void @{}({})\n  ret void\n",
                    extern_func.name, arg_list
                ));
            }
            _ => {
                wrapper.push_str(&format!(
                    "  %res = call {} @{}({})\n  ret {} %res\n",
                    ret_ty, extern_func.name, arg_list, ret_ty
                ));
            }
        }

        wrapper.push('}');
        Ok(wrapper)
    }
}

impl Default for FFICodegen {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extern_function_creation() {
        let func = ExternFunction::new("malloc".to_string(), NovaType::Float);
        assert_eq!(func.name, "malloc");
        assert_eq!(func.return_type, NovaType::Float);
    }

    #[test]
    fn extern_function_declaration() {
        let func = ExternFunction::new("sin".to_string(), NovaType::Float);
        let decl = func.emit_declaration();
        assert!(decl.contains("declare f64 @sin"));
    }

    #[test]
    fn ffi_codegen_create() {
        let codegen = FFICodegen::new();
        let _ = codegen;
    }

    #[test]
    fn ffi_codegen_wrapper_non_void() {
        let codegen = FFICodegen::new();
        let mut f = ExternFunction::new("sin".to_string(), NovaType::Float);
        f.add_parameter(NovaType::Float);
        let wrapper = codegen.generate_wrapper(&f).expect("wrapper");
        assert!(wrapper.contains("define f64 @sin_wrapper"));
        assert!(wrapper.contains("entry:"));
        assert!(wrapper.contains("call f64 @sin"));
        assert!(wrapper.contains("ret f64 %res"));
    }

    #[test]
    fn ffi_codegen_wrapper_void() {
        let codegen = FFICodegen::new();
        let mut f = ExternFunction::new("do_work".to_string(), NovaType::Void);
        f.add_parameter(NovaType::Int);
        let wrapper = codegen.generate_wrapper(&f).expect("wrapper");
        assert!(wrapper.contains("define void @do_work_wrapper"));
        assert!(wrapper.contains("call void @do_work"));
        assert!(wrapper.contains("ret void"));
    }
}
