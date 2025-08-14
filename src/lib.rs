//! Value Expr
//!
//! ## DSL
//!
//! ```dsl
//! Value:
//!     i32
//!     | Unary
//!     | Binary
//!     | Assign
//!     | Paren
//!     | FuncLike
//!     | Ident
//! Unary:
//!     - Value
//!     | ! Value
//! Binary:
//!     Value + Value
//!     | Value - Value
//!     | Value * Value
//!     | Value / Value
//!     | Value % Value
//!     | Value ^ Value
//!     | Value & Value
//!     | Value | Value
//!     | Value << Value
//!     | Value >> Value
//!     | Value == Value
//!     | Value != Value
//!     | Value > Value
//!     | Value < Value
//!     | Value >= Value
//!     | Value <= Value
//!     | Value && Value
//!     | Value || Value
//! Assign:
//!     Ident = Value
//!     | Ident += Value
//!     | Ident -= Value
//!     | Ident *= Value
//!     | Ident /= Value
//!     | Ident %= Value
//!     | Ident ^= Value
//!     | Ident &= Value
//!     | Ident |= Value
//!     | Ident <<= Value
//!     | Ident >>= Value
//! Paren:
//!     ( Values )
//! FuncLike:
//!     Ident ( Values )
//! Values:
//!     <nothing>
//!     | ValuesNext
//! ValuesNext:
//!     Value
//!     | Value , ValuesNext
//! Ident:
//!     <the rust lang ident>
//! ```
//!
//! ## Binary
//!
//! Here is the precedence of binary operators, going from strong to weak.
//! Operators at the same precedence level are all left-associative.
//!
//! If you're unsure about operator precedence, use parentheses to explicitly enforce the desired evaluation order.
//!
//! When mixing numerical and logical operations, any numeric value greater than 0 is considered true, otherwise false.
//! The logical value true corresponds to 1 numerically, while false corresponds to 0.
//!
//! | Operator | Remark |
//! | -------- | ------- |
//! | `*` `/` `%` | arithmetic |
//! | `+` `-` | arithmetic |
//! | `<<` `>>` | bit shift |
//! | `&` | bit and|
//! | `^` | bit xor |
//! | `\|` | bit or |
//! | `==` `!=` `>` `<` `>=` `<=` | compare |
//! | `&&` | logical and |
//! | `\|\|` | logical or |
//!
//! ## Assign
//!
//! Assign Operators all right-associative.
//!
//! ## FuncLike
//!
//! FuncLike, a function-like object, is similar to a function in that it takes multiple parameters.
//!
//! The difference is that it allows lazy evaluation, enabling you to freely control the order of evaluation.
//!

mod context;
mod data;
mod parser;
mod valuer;

pub use context::{ContextHelper, ContextHolder};
pub use data::{AssignOp, BinOp, UnOp, Value};
pub use valuer::{Context, Valued};
