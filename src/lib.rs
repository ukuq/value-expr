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
//! 运算优先级从低到高如下，同一优先级时按照顺序左结合运算。
//!
//! 在混合使用时，大于0的值被认为是true，否则为false，true对应的数值为1，false为0。
//!
//! ```x
//! 逻辑或	||
//! 逻辑与	&&
//! 数值比较	== != > < >= <=
//! 按位或	|
//! 按位异或	^
//! 按位与	&
//! 移位	<< >>
//! 数值运算	+-
//! 数值运算	*/%
//! ```
//!
//! ## FuncLike
//!
//! 类函数，与普通函数不同的是，它允许延迟计算。命名上，建议所有含延迟的函数均以_开头，普通函数以英文字母开头。
//!

mod context;
mod data;
mod parser;
mod valuer;

pub use context::{ContextHelper, ContextHolder};
pub use data::{AssignOp, BinOp, UnOp, Value};
pub use valuer::{Context, Valued};
