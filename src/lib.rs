pub enum UnOp {
    Not,
    Neg,
}
pub enum BinOp {
    /// +
    Add,
    /// -
    Sub,
    /// *
    Mul,
    /// /
    Div,
    /// %
    Rem,
    /// &&
    And,
    /// ||
    Or,
    /// ^
    BitXor,
    /// &
    BitAnd,
    /// |
    BitOr,
    /// <<
    Shl,
    /// >>
    Shr,
    /// ==
    Eq,
    /// <
    Lt,
    /// <=
    Le,
    /// !=
    Ne,
    /// >=
    Ge,
    /// >
    Gt,
}
pub enum Value {
    // 1
    Integer(i32),
    // -a
    Unary(UnOp, Box<Value>),
    // a+b
    Binary(BinOp, Box<Value>, Box<Value>),
    // (a)
    Paren(Vec<Value>),
    // a(b,c)
    FuncLike(String, Vec<Value>),
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
                input.parse::<Token![&&]>().map(|_| BinOp::And)
            } else if input.peek(Token![||]) {
                input.parse::<Token![||]>().map(|_| BinOp::Or)
            } else if input.peek(Token![<<]) {
                input.parse::<Token![<<]>().map(|_| BinOp::Shl)
            } else if input.peek(Token![>>]) {
                input.parse::<Token![>>]>().map(|_| BinOp::Shr)
            } else if input.peek(Token![==]) {
                input.parse::<Token![==]>().map(|_| BinOp::Eq)
            } else if input.peek(Token![<=]) {
                input.parse::<Token![<=]>().map(|_| BinOp::Le)
            } else if input.peek(Token![!=]) {
                input.parse::<Token![!=]>().map(|_| BinOp::Ne)
            } else if input.peek(Token![>=]) {
                input.parse::<Token![>=]>().map(|_| BinOp::Ge)
            } else if input.peek(Token![+]) {
                input.parse::<Token![+]>().map(|_| BinOp::Add)
            } else if input.peek(Token![-]) {
                input.parse::<Token![-]>().map(|_| BinOp::Sub)
            } else if input.peek(Token![*]) {
                input.parse::<Token![*]>().map(|_| BinOp::Mul)
            } else if input.peek(Token![/]) {
                input.parse::<Token![/]>().map(|_| BinOp::Div)
            } else if input.peek(Token![%]) {
                input.parse::<Token![%]>().map(|_| BinOp::Rem)
            } else if input.peek(Token![^]) {
                input.parse::<Token![^]>().map(|_| BinOp::BitXor)
            } else if input.peek(Token![&]) {
                input.parse::<Token![&]>().map(|_| BinOp::BitAnd)
            } else if input.peek(Token![|]) {
                input.parse::<Token![|]>().map(|_| BinOp::BitOr)
            } else if input.peek(Token![<]) {
                input.parse::<Token![<]>().map(|_| BinOp::Lt)
            } else if input.peek(Token![>]) {
                input.parse::<Token![>]>().map(|_| BinOp::Gt)
            } else {
                Err(input.error("expected binary operator"))
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
            let path: Ident = input.parse()?;
            if input.peek(token::Paren) {
                let content;
                parenthesized!(content in input);
                let data = content
                    .parse_terminated(Value::parse, Token![,])?
                    .into_iter()
                    .collect();
                return Ok(Value::FuncLike(path.to_string(), data));
            }
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
        }
    }
}
mod valuer {
    use super::*;
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

    pub trait IValue<T: ?Sized> {
        fn to_i32(&self, ctx: &mut T) -> i32;
    }
    pub trait IContext {
        fn call(&mut self, func: &str, args: &Vec<Value>) -> i32;
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

    impl<T: IContext, V: IValue<T>> IValue<T> for Vec<V> {
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
            }
        }
    }

    impl<T: IContext> IValue<T> for &Box<Value> {
        fn to_i32(&self, ctx: &mut T) -> i32 {
            self.as_ref().to_i32(ctx)
        }
    }

    impl IContext for () {
        fn call(&mut self, _func: &str, _args: &Vec<Value>) -> i32 {
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
                fn call(&mut self, func: &str, args: &Vec<Value>) -> i32 {
                    match func {
                        "add" => {
                            let args: Vec<_> = args.iter().map(|e| e.to_i32(self)).collect();
                            args[0] + args[1]
                        }
                        &_ => {
                            unreachable!()
                        }
                    }
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
pub use valuer::{IContext, IValue};
