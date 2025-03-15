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
//! ### _if 条件判断
//!
//! 计算第一个表达式为判断条件，为真时执行第二个表达式，否则执行第三个表达式，返回前依次执行剩余表达式。
//!
//! ```x
//! _if() =: 0
//! _if(a()) =: if a() { 0 } else { 0 }
//! _if(a(),b()) =: if a() { b() } else { 0 }
//! _if(a(),b(),c()) =: if a() { b() } else{ c() }
//! _if(a(),b(),c(),d()...) =: { let t = if a() { b() } else { c() }; d()...; t }
//! ```
//!
//! ### _fn 函数定义
//!
//! 取第一个参数为函数标识，第二个参数为函数体，剩余参数为函数参数，函数体中可以使用函数参数作为变量读写，返回值为函数ID。
//!
//! 若函数标识为Ident，则定义为函数名，否则为匿名函数计算其值作为函数ID。
//!
//! ```x
//! _fn(add,a+b,a,b) =: fn add(a,b) { a+b }
//! _if(1,a+b,a,b) =: |a,b|a+b
//! ```
//!
//! ### _call 函数调用
//!
//! 取第一个参数值为函数标识，剩余参数为函数参数，返回值为函数执行结果。
//!
//! 若函数标识为Ident，则取对应函数ID，否则计算其值作为函数ID。
//!
//! ```x
//! _fn(add,a+b,a,b)
//! _call(add,1,2) =: add(1,2)
//! _if(1,a+b,a,b)
//! _call(2-1,1,2) =: 1+2
//! ```
//!
//! ### _call_inline 匿名函数
//!
//! 取第一个参数值为函数体，剩余参数为函数参数，返回值为函数执行结果。
//!
//! ```x
//! _call_inline(arg0+arg1,1,2)
//! ```
//!
//! ### _scope 变量域
//!
//! 创建一个变量域，类似fork机制，变量域内可写入当前域，可读取当前域或父级域或父父级域等等，默认0，依次计算每个参数，取最后一个作为返回值，默认0。
//!
//! _call、_call_inline 函数调用同样会创建一个变量域。其余场景不会创建变量域。
//!
//! ```x
//! _scope(a=100,_assert(a==100)),
//! _scope(a=100,_scope(_assert(a==100))),
//! _scope(a=100,a=200,_assert(a==200)),
//! _scope(a=100,_scope(a=200),_assert(a==100))
//! _fn(f1,_assert(a==0)),
//! _scope(a=100,_fn(f1,_assert(a==100))),
//! _scope(a=100,_fn(f1,(a=200,_assert(a==200))),_assert(a==100))
//! ```
//!
//! ### _while 循环
//!
//! 取第一个参数值为循环条件，循环依次计算每个参数，取最后一次循环最后一个值作为返回值。
//!
//! ```x
//! (
//! i=10,
//! _while(i<20,
//!     _if(i%2==0,_log(i_is,i)),
//!     i+=1,
//!     i
//! )
//! )
//! ```
//!
//! ### _log、_debug、_assert 调试工具
//!
//! _log 日志打印，保留第一个参数，打印剩余参数的值，返回最后一个值的结果。
//!
//! _debug 打印内部数据状态，参数、返回值无意义。
//!
//! _assert 断言第一个参数，剩余参数、返回值无意义。
//!

#[derive(Debug, Clone)]
pub enum UnOp {
    Not,
    Neg,
}
#[derive(Debug, Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    BitXor,
    BitAnd,
    BitOr,
    Shl,
    Shr,
    Eq,
    Lt,
    Le,
    Ne,
    Ge,
    Gt,
}
#[derive(Debug, Clone)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    BitXorAssign,
    BitAndAssign,
    BitOrAssign,
    ShlOrAssign,
    ShrOrAssign,
}
#[derive(Debug, Clone)]
pub enum Value {
    /// 1
    Integer(i32),
    /// -a
    Unary(UnOp, Box<Value>),
    /// a+b
    Binary(BinOp, Box<Value>, Box<Value>),
    /// (a)
    Paren(Vec<Value>),
    /// a(b,c)
    FuncLike(String, Vec<Value>),
    /// a
    Ident(String),
    /// a=1
    Assign(AssignOp, String, Box<Value>),
    /// << inner use >>
    Copy(Arc<Value>),
}

macro_rules! i2b {
    ($expr:expr) => {
        $expr > 0
    };
}
macro_rules! b2i {
    ($expr:expr) => {
        if $expr {
            1
        } else {
            0
        }
    };
}

mod parser {
    use super::*;
    use proc_macro2::TokenStream;
    use std::str::FromStr;
    use syn::parse::discouraged::Speculative;
    use syn::parse::{Parse, ParseStream};
    use syn::{parenthesized, token, Ident, LitInt, Token};

    #[derive(Eq, PartialEq, Ord, PartialOrd)]
    enum Precedence {
        Any,
        Or,
        And,
        Compare,
        BitOr,
        BitXor,
        BitAnd,
        Shift,
        Arithmetic,
        Term,
    }

    impl Precedence {
        fn of(op: &BinOp) -> Self {
            match op {
                BinOp::Add | BinOp::Sub => Precedence::Arithmetic,
                BinOp::Mul | BinOp::Div | BinOp::Rem => Precedence::Term,
                BinOp::And => Precedence::And,
                BinOp::Or => Precedence::Or,
                BinOp::BitXor => Precedence::BitXor,
                BinOp::BitAnd => Precedence::BitAnd,
                BinOp::BitOr => Precedence::BitOr,
                BinOp::Shl | BinOp::Shr => Precedence::Shift,
                BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
                    Precedence::Compare
                }
            }
        }
    }

    impl Parse for UnOp {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let lookahead = input.lookahead1();
            if lookahead.peek(Token![!]) {
                input.parse::<Token![!]>().map(|_| Self::Not)
            } else if lookahead.peek(Token![-]) {
                input.parse::<Token![-]>().map(|_| Self::Neg)
            } else {
                Err(input.error("expected unary operator"))
            }
        }
    }

    impl Parse for BinOp {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            if input.peek(Token![&&]) {
                input.parse::<Token![&&]>().map(|_| Self::And)
            } else if input.peek(Token![||]) {
                input.parse::<Token![||]>().map(|_| Self::Or)
            } else if input.peek(Token![<<]) {
                input.parse::<Token![<<]>().map(|_| Self::Shl)
            } else if input.peek(Token![>>]) {
                input.parse::<Token![>>]>().map(|_| Self::Shr)
            } else if input.peek(Token![==]) {
                input.parse::<Token![==]>().map(|_| Self::Eq)
            } else if input.peek(Token![<=]) {
                input.parse::<Token![<=]>().map(|_| Self::Le)
            } else if input.peek(Token![!=]) {
                input.parse::<Token![!=]>().map(|_| Self::Ne)
            } else if input.peek(Token![>=]) {
                input.parse::<Token![>=]>().map(|_| Self::Ge)
            } else if input.peek(Token![+]) {
                input.parse::<Token![+]>().map(|_| Self::Add)
            } else if input.peek(Token![-]) {
                input.parse::<Token![-]>().map(|_| Self::Sub)
            } else if input.peek(Token![*]) {
                input.parse::<Token![*]>().map(|_| Self::Mul)
            } else if input.peek(Token![/]) {
                input.parse::<Token![/]>().map(|_| Self::Div)
            } else if input.peek(Token![%]) {
                input.parse::<Token![%]>().map(|_| Self::Rem)
            } else if input.peek(Token![^]) {
                input.parse::<Token![^]>().map(|_| Self::BitXor)
            } else if input.peek(Token![&]) {
                input.parse::<Token![&]>().map(|_| Self::BitAnd)
            } else if input.peek(Token![|]) {
                input.parse::<Token![|]>().map(|_| Self::BitOr)
            } else if input.peek(Token![<]) {
                input.parse::<Token![<]>().map(|_| Self::Lt)
            } else if input.peek(Token![>]) {
                input.parse::<Token![>]>().map(|_| Self::Gt)
            } else {
                Err(input.error("expected binary operator"))
            }
        }
    }

    impl Parse for AssignOp {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            if input.peek(Token![=]) {
                input.parse::<Token![=]>().map(|_| Self::Assign)
            } else if input.peek(Token![+=]) {
                input.parse::<Token![+=]>().map(|_| Self::AddAssign)
            } else if input.peek(Token![-=]) {
                input.parse::<Token![-=]>().map(|_| Self::SubAssign)
            } else if input.peek(Token![*=]) {
                input.parse::<Token![*=]>().map(|_| Self::MulAssign)
            } else if input.peek(Token![/=]) {
                input.parse::<Token![/=]>().map(|_| Self::DivAssign)
            } else if input.peek(Token![%=]) {
                input.parse::<Token![%=]>().map(|_| Self::RemAssign)
            } else if input.peek(Token![^=]) {
                input.parse::<Token![^=]>().map(|_| Self::BitXorAssign)
            } else if input.peek(Token![&=]) {
                input.parse::<Token![&=]>().map(|_| Self::BitAndAssign)
            } else if input.peek(Token![|=]) {
                input.parse::<Token![|=]>().map(|_| Self::BitOrAssign)
            } else if input.peek(Token![<<=]) {
                input.parse::<Token![<<=]>().map(|_| Self::ShlOrAssign)
            } else if input.peek(Token![>>=]) {
                input.parse::<Token![>>=]>().map(|_| Self::ShrOrAssign)
            } else {
                Err(input.error("expected assign operator"))
            }
        }
    }

    impl Parse for Value {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let lhs = unary_value(input)?;
            parse_value(input, lhs, Precedence::Any)
        }
    }

    fn unary_value(input: ParseStream) -> syn::Result<Value> {
        if input.peek(Token![!]) || input.peek(Token![-]) {
            Ok(Value::Unary(input.parse()?, Box::new(unary_value(input)?)))
        } else {
            atom_value(input)
        }
    }

    fn atom_value(input: ParseStream) -> syn::Result<Value> {
        if input.peek(token::Paren) {
            let content;
            parenthesized!(content in input);
            let data = content
                .parse_terminated(Value::parse, Token![,])?
                .into_iter()
                .collect();
            return Ok(Value::Paren(data));
        }
        if input.peek(LitInt) {
            let integer = input.parse::<LitInt>()?.base10_parse::<i32>()?;
            return Ok(Value::Integer(integer));
        }
        if input.peek(Ident) {
            let ident = input.parse::<Ident>()?.to_string();
            if input.peek(token::Paren) {
                let content;
                parenthesized!(content in input);
                let data = content
                    .parse_terminated(Value::parse, Token![,])?
                    .into_iter()
                    .collect();
                return Ok(Value::FuncLike(ident, data));
            }
            if !input.peek(Token![==]) {
                let ahead = input.fork();
                if let Ok(op) = ahead.parse::<AssignOp>() {
                    input.advance_to(&ahead);
                    return Ok(Value::Assign(op, ident, input.parse()?));
                }
            }
            return Ok(Value::Ident(ident));
        }
        Err(input.lookahead1().error())
    }

    fn peek_precedence(input: ParseStream) -> Precedence {
        if let Ok(op) = input.fork().parse() {
            Precedence::of(&op)
        } else {
            Precedence::Any
        }
    }

    fn parse_value(input: ParseStream, mut lhs: Value, base: Precedence) -> syn::Result<Value> {
        loop {
            let ahead = input.fork();
            if let Some(op) = match ahead.parse::<BinOp>() {
                Ok(op) if Precedence::of(&op) >= base => Some(op),
                _ => None,
            } {
                input.advance_to(&ahead);
                let precedence = Precedence::of(&op);
                let mut rhs = unary_value(input)?;
                loop {
                    let next = peek_precedence(input);
                    if next > precedence {
                        rhs = parse_value(input, rhs, next)?;
                    } else {
                        break;
                    }
                }
                lhs = Value::Binary(op, Box::new(lhs), Box::new(rhs));
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    impl Value {
        pub fn parse_str(input: &str) -> syn::Result<Self> {
            syn::parse2(TokenStream::from_str(input)?)
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::Value;

        #[test]
        fn test() {
            let test = |e| Value::parse_str(e).unwrap();
            test("1");
            test("-1");
            test("1+2");
            test("(1,2,3)");
            test("(1+2,3,a(1))");
            test("a");
            test("a+=1");
        }
    }
}
mod valuer {
    use super::*;

    pub trait IValue<T: ?Sized> {
        fn to_i32(&self, ctx: &mut T) -> i32;
    }
    pub trait IContext {
        fn call(&mut self, func: &str, values: &Vec<Value>) -> i32;
        fn ident_get(&self, ident: &str) -> i32;
        fn ident_set(&mut self, ident: &str, value: i32);
    }

    impl UnOp {
        pub fn to_i32<T: IContext, V: IValue<T>>(&self, ctx: &mut T, value: V) -> i32 {
            match self {
                UnOp::Not => b2i!(!i2b!(value.to_i32(ctx))),
                UnOp::Neg => -value.to_i32(ctx),
            }
        }
    }

    impl BinOp {
        pub fn to_i32<T: IContext, V: IValue<T>>(&self, ctx: &mut T, left: V, right: V) -> i32 {
            match self {
                BinOp::Add => left.to_i32(ctx) + right.to_i32(ctx),
                BinOp::Sub => left.to_i32(ctx) - right.to_i32(ctx),
                BinOp::Mul => left.to_i32(ctx) * right.to_i32(ctx),
                BinOp::Div => left.to_i32(ctx) / right.to_i32(ctx),
                BinOp::Rem => left.to_i32(ctx) % right.to_i32(ctx),
                BinOp::BitXor => left.to_i32(ctx) ^ right.to_i32(ctx),
                BinOp::BitAnd => left.to_i32(ctx) & right.to_i32(ctx),
                BinOp::BitOr => left.to_i32(ctx) | right.to_i32(ctx),
                BinOp::Shl => left.to_i32(ctx) << right.to_i32(ctx),
                BinOp::Shr => left.to_i32(ctx) >> right.to_i32(ctx),
                BinOp::And => b2i!(i2b!(left.to_i32(ctx)) && i2b!(right.to_i32(ctx))),
                BinOp::Or => b2i!(i2b!(left.to_i32(ctx)) || i2b!(right.to_i32(ctx))),
                BinOp::Eq => b2i!(left.to_i32(ctx) == right.to_i32(ctx)),
                BinOp::Lt => b2i!(left.to_i32(ctx) < right.to_i32(ctx)),
                BinOp::Le => b2i!(left.to_i32(ctx) <= right.to_i32(ctx)),
                BinOp::Ne => b2i!(left.to_i32(ctx) != right.to_i32(ctx)),
                BinOp::Ge => b2i!(left.to_i32(ctx) >= right.to_i32(ctx)),
                BinOp::Gt => b2i!(left.to_i32(ctx) > right.to_i32(ctx)),
            }
        }
    }

    impl AssignOp {
        pub fn to_i32<T: IContext, V: IValue<T>>(&self, ctx: &mut T, ident: &str, value: V) -> i32 {
            let v = match self {
                AssignOp::Assign => value.to_i32(ctx),
                AssignOp::AddAssign => ctx.ident_get(ident) + value.to_i32(ctx),
                AssignOp::SubAssign => ctx.ident_get(ident) - value.to_i32(ctx),
                AssignOp::MulAssign => ctx.ident_get(ident) * value.to_i32(ctx),
                AssignOp::DivAssign => ctx.ident_get(ident) / value.to_i32(ctx),
                AssignOp::RemAssign => ctx.ident_get(ident) % value.to_i32(ctx),
                AssignOp::BitXorAssign => ctx.ident_get(ident) ^ value.to_i32(ctx),
                AssignOp::BitAndAssign => ctx.ident_get(ident) & value.to_i32(ctx),
                AssignOp::BitOrAssign => ctx.ident_get(ident) | value.to_i32(ctx),
                AssignOp::ShlOrAssign => ctx.ident_get(ident) << value.to_i32(ctx),
                AssignOp::ShrOrAssign => ctx.ident_get(ident) >> value.to_i32(ctx),
            };
            ctx.ident_set(ident, v);
            v
        }
    }

    impl<T: IContext, V: IValue<T>> IValue<T> for [V] {
        fn to_i32(&self, ctx: &mut T) -> i32 {
            let mut last = 0;
            for value in self.iter() {
                last = value.to_i32(ctx);
            }
            last
        }
    }

    impl<T: IContext> IValue<T> for Value {
        fn to_i32(&self, ctx: &mut T) -> i32 {
            match self {
                Value::Integer(v) => *v,
                Value::Unary(op, v) => op.to_i32(ctx, v),
                Value::Binary(op, l, r) => op.to_i32(ctx, l, r),
                Value::Paren(v) => v.to_i32(ctx),
                Value::FuncLike(v, args) => ctx.call(v, args),
                Value::Ident(ident) => ctx.ident_get(ident),
                Value::Assign(op, ident, v) => op.to_i32(ctx, ident, v),
                Value::Copy(v) => v.to_i32(ctx),
            }
        }
    }

    impl<T: IContext> IValue<T> for &Box<Value> {
        fn to_i32(&self, ctx: &mut T) -> i32 {
            self.as_ref().to_i32(ctx)
        }
    }

    impl IContext for () {
        fn call(&mut self, _func: &str, _values: &Vec<Value>) -> i32 {
            unreachable!()
        }

        fn ident_get(&self, _ident: &str) -> i32 {
            unreachable!()
        }

        fn ident_set(&mut self, _ident: &str, _value: i32) {
            unreachable!()
        }
    }

    impl Value {
        fn optimize_value(&self) -> Option<i32> {
            match self {
                Value::Integer(v) => Some(*v),
                Value::Unary(op, v) => {
                    Some(op.to_i32(&mut (), Value::Integer(v.optimize_value()?)))
                }
                Value::Binary(op, l, r) => {
                    let left = l.optimize_value()?;
                    match op {
                        BinOp::And => {
                            if !i2b!(left) {
                                return Some(b2i!(false));
                            }
                        }
                        BinOp::Or => {
                            if i2b!(left) {
                                return Some(b2i!(true));
                            }
                        }
                        _ => {}
                    }
                    Some(op.to_i32(
                        &mut (),
                        Value::Integer(left),
                        Value::Integer(r.optimize_value()?),
                    ))
                }
                Value::Paren(v) => {
                    let mut res = 0;
                    for x in v {
                        res = x.optimize_value()?
                    }
                    Some(res)
                }
                Value::FuncLike(_, _) => None,
                Value::Ident(_) => None,
                Value::Assign(_, _, _) => None,
                Value::Copy(_) => None,
            }
        }
        pub fn optimize(self) -> Self {
            match self.optimize_value() {
                None => self,
                Some(v) => Self::Integer(v),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::valuer::{IContext, IValue};
        use crate::Value;

        #[test]
        fn test() {
            struct C;
            impl IContext for C {
                fn call(&mut self, func: &str, values: &Vec<Value>) -> i32 {
                    match func {
                        "add" => {
                            let args: Vec<_> = values.iter().map(|e| e.to_i32(self)).collect();
                            args[0] + args[1]
                        }
                        &_ => {
                            unreachable!()
                        }
                    }
                }

                fn ident_get(&self, _ident: &str) -> i32 {
                    unreachable!()
                }

                fn ident_set(&mut self, _ident: &str, _value: i32) {
                    unreachable!()
                }
            }
            impl C {
                fn test(&mut self, str: &str) -> i32 {
                    Value::parse_str(str).unwrap().to_i32(self)
                }
            }
            let mut c = C {};
            let ctx = &mut c;
            assert_eq!(ctx.test("1+1"), 2);
            assert_eq!(ctx.test("add(1,2*5)"), 11);
        }
    }
}
mod context {
    use crate::{IContext, IValue, Value};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[derive(Debug)]
    pub struct FnDef {
        #[allow(dead_code)]
        pub fid: i32,
        pub func: Value,
        pub params: HashMap<String, i32>,
    }
    #[derive(Debug)]
    pub struct FnFrame {
        pub func: Arc<FnDef>,
        pub args: Vec<i32>,
    }
    #[derive(Debug)]
    pub struct VarFrame {
        pub age: i32,
        pub value: i32,
    }
    #[derive(Default)]
    pub struct ContextHelper {
        pointer: i32,
        array0: Vec<i32>,
        array_map: HashMap<i32, Vec<i32>>,
        fn_map: HashMap<i32, Arc<FnDef>>,
        fn_name: HashMap<String, i32>,
        fn_stack: Vec<FnFrame>,
        var_stack: HashMap<String, Vec<VarFrame>>,
    }
    pub trait IContextHelper: Sized {
        fn ctx(&mut self) -> &mut ContextHelper;
        fn ctx_ref(&self) -> &ContextHelper;
        fn ctx_log(&self, msg: &str);
        #[doc(hidden)]
        fn next_key(&mut self, key: i32) -> i32 {
            if key > 0 {
                key
            } else {
                let ctx = self.ctx();
                ctx.pointer += 1;
                ctx.pointer
            }
        }
        #[doc(hidden)]
        fn scope_begin(&mut self) -> i32 {
            let ctx = self.ctx();
            let point = ctx.pointer;
            ctx.pointer += 1;
            point
        }
        #[doc(hidden)]
        fn scope_end(&mut self, point: i32) {
            self.ctx().pointer = point;
            for (_, frames) in &mut self.ctx().var_stack {
                while let Some(frame) = frames.last_mut() {
                    if frame.age > point {
                        frames.pop();
                    } else {
                        break;
                    }
                }
            }
        }
        #[doc(hidden)]
        fn array_def(&mut self, key: i32, elements: Vec<i32>) -> i32 {
            let key = self.next_key(key);
            self.ctx().array_map.insert(key, elements);
            key
        }
        #[doc(hidden)]
        fn array_get(&self, key: i32) -> &Vec<i32> {
            let ctx = self.ctx_ref();
            match ctx.array_map.get(&key) {
                Some(v) => v,
                None => &ctx.array0,
            }
        }
        #[doc(hidden)]
        fn fn_def(&mut self, fid: i32, func: &Value, params: HashMap<String, i32>) -> i32 {
            let fid = self.next_key(fid);
            self.ctx().fn_map.insert(
                fid,
                Arc::new(FnDef {
                    fid,
                    func: func.clone(),
                    params,
                }),
            );
            fid
        }
        #[doc(hidden)]
        fn fn_call(&mut self, fid: i32, args: Vec<i32>) -> i32 {
            let point = self.scope_begin();
            let func = self.ctx().fn_map.get(&fid).unwrap().clone();
            self.ctx().fn_stack.push(FnFrame {
                func: func.clone(),
                args,
            });
            let res = func.func.to_i32(self);
            self.ctx().fn_stack.pop();
            self.scope_end(point);
            res
        }
        #[doc(hidden)]
        fn _if(&mut self, args: &Vec<Value>) -> i32 {
            match args.len() {
                0 => 0,
                1 => {
                    if i2b!(args[0].to_i32(self)) {
                        0
                    } else {
                        0
                    }
                }
                2 => {
                    if i2b!(args[0].to_i32(self)) {
                        args[1].to_i32(self)
                    } else {
                        0
                    }
                }
                _ => {
                    let res = if i2b!(args[0].to_i32(self)) {
                        args[1].to_i32(self)
                    } else {
                        args[2].to_i32(self)
                    };
                    args[3..].to_i32(self);
                    res
                }
            }
        }
        #[doc(hidden)]
        fn _fn(&mut self, args: &Vec<Value>) -> i32 {
            let fid = if let Value::Ident(ident) = &args[0] {
                let fid = self.next_key(0);
                self.ctx().fn_name.insert(ident.clone(), fid);
                fid
            } else {
                args[0].to_i32(self)
            };
            let mut params = HashMap::new();
            for idx in 2..args.len() {
                if let Value::Ident(ident) = &args[idx] {
                    params.insert(ident.clone(), idx as i32 - 2);
                }
            }
            self.fn_def(fid, &args[1], params);
            fid
        }
        #[doc(hidden)]
        fn _call(&mut self, args: &Vec<Value>) -> i32 {
            let fid = if let Value::Ident(ident) = &args[0] {
                self.ctx().fn_name.get(ident).map(|e| *e).unwrap()
            } else {
                args[0].to_i32(self)
            };
            let args = args[1..].to_i32_vec(self);
            self.fn_call(fid, args)
        }
        #[doc(hidden)]
        fn _call_inline(&mut self, args: &Vec<Value>) -> i32 {
            match args.len() {
                0 => 0,
                1 => args[0].to_i32(self),
                _ => {
                    let fid = self.fn_def(0, &args[0], HashMap::new());
                    let args = args[1..].to_i32_vec(self);
                    self.fn_call(fid, args)
                }
            }
        }
        #[doc(hidden)]
        fn _scope(&mut self, args: &Vec<Value>) -> i32 {
            let point = self.scope_begin();
            let res = args.to_i32(self);
            self.scope_end(point);
            res
        }
        #[doc(hidden)]
        fn _while(&mut self, args: &Vec<Value>) -> i32 {
            let mut res = 0;
            while i2b!(args[0].to_i32(self)) {
                res = args[1..].to_i32(self);
            }
            res
        }
        #[doc(hidden)]
        fn _log(&mut self, args: &Vec<Value>) -> i32 {
            let msg = if let Some(Value::Ident(ident)) = args.first() {
                ident
            } else {
                "<<value>>"
            };
            let args = args[1..].to_i32_vec(self);
            self.ctx_log(&format!("{} {:?}", msg, args));
            args.last().map(|e| *e).unwrap_or(0)
        }
        #[doc(hidden)]
        fn _assert(&mut self, args: &Vec<Value>) -> i32 {
            assert!(i2b!(args[0].to_i32(self)));
            1
        }
        #[doc(hidden)]
        fn _debug(&mut self, _args: &Vec<Value>) -> i32 {
            let msg = format!("var stack:{:?}", self.ctx().var_stack);
            self.ctx_log(&msg);
            0
        }
    }
    impl<T: IContextHelper> IContext for T {
        fn call(&mut self, func: &str, values: &Vec<Value>) -> i32 {
            match func {
                "_if" => self._if(values),
                "_fn" => self._fn(values),
                "_call" => self._call(values),
                "_call_inline" => self._call_inline(values),
                "_scope" => self._scope(values),
                "_while" => self._while(values),
                "_log" => self._log(values),
                "_assert" => self._assert(values),
                "_debug" => self._debug(values),
                _ => {
                    let args = values.to_i32_vec(self);
                    match self.ctx().fn_name.get(func).map(|e| *e) {
                        Some(fid) => self.fn_call(fid, args),
                        None => {
                            unreachable!("unknown function: {}", func);
                        }
                    }
                }
            }
        }

        fn ident_get(&self, ident: &str) -> i32 {
            if let Some(FnFrame { func, args }) = self.ctx_ref().fn_stack.last() {
                if let Some(idx) = func.params.get(ident) {
                    return args.get(*idx as usize).map(|e| *e).unwrap_or(0);
                }
            }
            if let Some(fid) = self.ctx_ref().fn_name.get(ident) {
                return *fid;
            }
            if let Some(vec) = self.ctx_ref().var_stack.get(ident) {
                let age = self.ctx_ref().pointer;
                if let Some(frame) = vec.iter().rev().find(|frame| frame.age <= age) {
                    return frame.value;
                }
            }
            0
        }

        fn ident_set(&mut self, ident: &str, value: i32) {
            if let Some(FnFrame { func, args }) = self.ctx().fn_stack.last_mut() {
                if let Some(idx) = func.params.get(ident) {
                    let idx = *idx as usize;
                    for _ in args.len()..=idx {
                        args.push(0);
                    }
                    args[idx] = value;
                    return;
                }
            }
            let ctx = self.ctx();
            let age = ctx.pointer;
            let frames = ctx.var_stack.entry(ident.to_string()).or_insert(vec![]);
            while let Some(frame) = frames.last_mut() {
                if frame.age > age {
                    frames.pop();
                } else if frame.age == age {
                    frame.value = value;
                    return;
                } else {
                    break;
                }
            }
            frames.push(VarFrame { age, value });
        }
    }

    trait IValueVec<T> {
        fn to_i32_vec(&self, ctx: &mut T) -> Vec<i32>;
    }
    impl<T: IContext, V: IValue<T>> IValueVec<T> for [V] {
        fn to_i32_vec(&self, ctx: &mut T) -> Vec<i32> {
            self.iter().map(|v| v.to_i32(ctx)).collect()
        }
    }

    #[derive(Default)]
    pub struct DemoContext {
        ctx: ContextHelper,
    }
    impl DemoContext {
        pub fn exec(&mut self, str: &str) {
            let v = Value::parse_str(str).unwrap().to_i32(self);
            println!("exec_value_is: {}", v);
        }
    }
    impl IContextHelper for DemoContext {
        fn ctx(&mut self) -> &mut ContextHelper {
            &mut self.ctx
        }

        fn ctx_ref(&self) -> &ContextHelper {
            &self.ctx
        }

        fn ctx_log(&self, msg: &str) {
            println!("{}", msg);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn test() {
            let mut ctx = DemoContext::default();
            //条件
            ctx.exec(
                "(
            _assert(_if(1,2,3)==2),
            _assert(_if(-1,2,3)==3),
            )",
            );
            //函数
            ctx.exec(
                "(
            _fn(add,a+b,a,b),
            _assert(add(1)==1),
            _assert(add(1,2)==3),
            _assert(_call(add,1,2)==3)
            )",
            );
            //循环
            ctx.exec(
                "_log(_while,
            i=10,
            _while(i<100000,
                _if(i%10000==0,_log(i_is,i)),
                i+=1,
                i
            )
            )",
            );
            //递归
            ctx.exec(
                "(
            _fn(fib1,_if(n<2,a2,fib1(n-1,a2,a1+a2)),n,a1,a2),
            _fn(fib,fib1(n,1,1),n),
            _log(fib,fib(0),fib(1),fib(2),fib(3),fib(10),fib(19)),
            _assert(6765==fib(19))
            )",
            );
            //作用域
            ctx.exec(
                "(
            _scope(a=100,_assert(a==100)),
            _scope(a=100,_scope(_assert(a==100))),
            _scope(a=100,a=200,_assert(a==200)),
            _scope(a=100,_scope(a=200),_assert(a==100)),
            _fn(f1,_assert(a==0)),
            _scope(a=100,_fn(f1,_assert(a==100))),
            _scope(a=100,_fn(f1,(a=200,_assert(a==200))),_assert(a==100))
            )",
            );
        }
    }
}

pub use context::{ContextHelper, DemoContext, IContextHelper};
use std::sync::Arc;
pub use valuer::{IContext, IValue};
