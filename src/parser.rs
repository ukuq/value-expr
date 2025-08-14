use super::data::*;
use proc_macro2::TokenStream;
use std::str::FromStr;
use syn::ext::IdentExt;
use syn::parse::discouraged::Speculative;
use syn::parse::{Parse, ParseStream};
use syn::{parenthesized, token, Ident, LitInt, Token};

macro_rules! op {
    ($name:ident => $msg:literal; $([$($tt:tt)+] => $ident:ident);+$(;)?) => {
        impl Parse for $name {
            fn parse(input: ParseStream) -> syn::Result<Self> {
                let lookahead = input.lookahead1();
                $(if lookahead.peek(Token![$($tt)+]) {
                    input.parse::<Token![$($tt)+]>().map(|_|Self::$ident)
                } else)+ {
                    Err(input.error($msg))
                }
            }
        }
        impl $name {
            #[allow(unused)]
            pub fn desc(self) -> &'static str {
                match self {
                    $(Self::$ident => stringify!($($tt)+)),+
                }
            }
        }
    };
}

op! {
    UnOp => "expected unary operator";
    [!] => Not;
    [-] => Neg;
}

op! {
    BinOp => "expected binary operator";
    [&&] => And;
    [||] => Or;
    [<<] => Shl;
    [>>] => Shr;
    [==] => Eq;
    [<=] => Le;
    [!=] => Ne;
    [>=] => Ge;
    [+] => Add;
    [-] => Sub;
    [*] => Mul;
    [/] => Div;
    [%] => Rem;
    [^] => BitXor;
    [&] => BitAnd;
    [|] => BitOr;
    [<] => Lt;
    [>] => Gt;
}

op! {
    AssignOp => "expected assignment operator";
    [=] => Assign;
    [+=] => AddAssign;
    [-=] => SubAssign;
    [*=] => MulAssign;
    [/=] => DivAssign;
    [%=] => RemAssign;
    [^=] => BitAndAssign;
    [&=] => BitOrAssign;
    [|=] => BitXorAssign;
    [<<=] => ShlAssign;
    [>>=] => ShrAssign;
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
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
    if input.peek(Ident::peek_any) {
        let ident = input.call(Ident::parse_any)?.to_string();
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

impl Parse for Value {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lhs = unary_value(input)?;
        parse_value(input, lhs, Precedence::Any)
    }
}

impl Value {
    pub fn parse_str(input: &str) -> syn::Result<Self> {
        syn::parse2(TokenStream::from_str(input)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Deref;

    #[test]
    fn test_basic_parsing() {
        let test = |e| Value::parse_str(e).unwrap();
        test("1");
        test("-1");
        test("1+2");
        test("(1,2,3)");
        test("(1+2,3,a(1))");
        test("a");
        test("a+=1");
    }

    #[test]
    fn test_unary_operators() {
        // 测试一元操作符
        let neg_result = Value::parse_str("-42").unwrap();
        match neg_result {
            Value::Unary(UnOp::Neg, val) => match val.as_ref() {
                Value::Integer(42) => (),
                _ => panic!("Expected integer 42"),
            },
            _ => panic!("Expected unary negation"),
        }

        let not_result = Value::parse_str("!true_var").unwrap();
        match not_result {
            Value::Unary(UnOp::Not, val) => match val.as_ref() {
                Value::Ident(ref name) => assert_eq!(name, "true_var"),
                _ => panic!("Expected identifier"),
            },
            _ => panic!("Expected unary not"),
        }

        // 测试嵌套一元操作符
        let double_neg = Value::parse_str("--5").unwrap();
        match double_neg {
            Value::Unary(UnOp::Neg, val) => match val.as_ref() {
                Value::Unary(UnOp::Neg, ref inner) => match **inner {
                    Value::Integer(5) => (),
                    _ => panic!("Expected integer 5"),
                },
                _ => panic!("Expected nested negation"),
            },
            _ => panic!("Expected outer negation"),
        }
    }

    #[test]
    fn test_binary_operators() {
        // 测试算术操作�?
        let add = Value::parse_str("3 + 5").unwrap();
        match add {
            Value::Binary(BinOp::Add, left, right) => {
                assert!(matches!(*left, Value::Integer(3)));
                assert!(matches!(*right, Value::Integer(5)));
            }
            _ => panic!("Expected addition"),
        }

        let mul = Value::parse_str("2 * 4").unwrap();
        match mul {
            Value::Binary(BinOp::Mul, left, right) => {
                assert!(matches!(*left, Value::Integer(2)));
                assert!(matches!(*right, Value::Integer(4)));
            }
            _ => panic!("Expected multiplication"),
        }

        // 测试比较操作�?
        let eq = Value::parse_str("a == b").unwrap();
        match eq {
            Value::Binary(BinOp::Eq, left, right) => {
                assert!(matches!(*left, Value::Ident(ref name) if name == "a"));
                assert!(matches!(*right, Value::Ident(ref name) if name == "b"));
            }
            _ => panic!("Expected equality"),
        }

        // 测试位操作符
        let bit_and = Value::parse_str("x & y").unwrap();
        match bit_and {
            Value::Binary(BinOp::BitAnd, left, right) => {
                assert!(matches!(*left, Value::Ident(ref name) if name == "x"));
                assert!(matches!(*right, Value::Ident(ref name) if name == "y"));
            }
            _ => panic!("Expected bitwise and"),
        }

        // 测试逻辑操作�?
        let logical_or = Value::parse_str("a || b").unwrap();
        match logical_or {
            Value::Binary(BinOp::Or, left, right) => {
                assert!(matches!(*left, Value::Ident(ref name) if name == "a"));
                assert!(matches!(*right, Value::Ident(ref name) if name == "b"));
            }
            _ => panic!("Expected logical or"),
        }
    }

    #[test]
    fn test_operator_precedence() {
        // 测试操作符优先级
        let expr = Value::parse_str("1 + 2 * 3").unwrap();
        match expr {
            Value::Binary(BinOp::Add, left, right) => {
                assert!(matches!(*left, Value::Integer(1)));
                match *right {
                    Value::Binary(BinOp::Mul, ref left, ref right) => {
                        assert!(matches!(**left, Value::Integer(2)));
                        assert!(matches!(**right, Value::Integer(3)));
                    }
                    _ => panic!("Expected multiplication to have higher precedence"),
                }
            }
            _ => panic!("Expected addition as root"),
        }

        // 测试比较与算术的优先�?
        let expr2 = Value::parse_str("a + b == c * d").unwrap();
        match expr2 {
            Value::Binary(BinOp::Eq, left, right) => match (*left, *right) {
                (Value::Binary(BinOp::Add, _, _), Value::Binary(BinOp::Mul, _, _)) => (),
                _ => panic!(
                    "Expected arithmetic operations to have higher precedence than comparison"
                ),
            },
            _ => panic!("Expected equality as root"),
        }
    }

    #[test]
    fn test_assignment_operators() {
        // 测试各种赋值操作符
        let simple_assign = Value::parse_str("x = 10").unwrap();
        match simple_assign {
            Value::Assign(AssignOp::Assign, ref ident, ref val) => {
                assert_eq!(ident, "x");
                assert!(matches!(*val.as_ref(), Value::Integer(10)));
            }
            _ => panic!("Expected simple assignment"),
        }

        let add_assign = Value::parse_str("counter += 5").unwrap();
        match add_assign {
            Value::Assign(AssignOp::AddAssign, ref ident, ref val) => {
                assert_eq!(ident, "counter");
                assert!(matches!(*val.as_ref(), Value::Integer(5)));
            }
            _ => panic!("Expected add assignment"),
        }

        let mul_assign = Value::parse_str("result *= factor").unwrap();
        match mul_assign {
            Value::Assign(AssignOp::MulAssign, ref ident, ref val) => {
                assert_eq!(ident, "result");
                assert!(matches!(*val.as_ref(), Value::Ident(ref name) if name == "factor"));
            }
            _ => panic!("Expected multiply assignment"),
        }

        let shl_assign = Value::parse_str("bits <<= shift").unwrap();
        match shl_assign {
            Value::Assign(AssignOp::ShlAssign, ref ident, ref val) => {
                assert_eq!(ident, "bits");
                assert!(matches!(*val.as_ref(), Value::Ident(ref name) if name == "shift"));
            }
            _ => panic!("Expected shift left assignment"),
        }
    }

    #[test]
    fn test_parentheses() {
        // 测试括号表达�?
        let simple_paren = Value::parse_str("(42)").unwrap();
        match simple_paren {
            Value::Paren(ref values) => {
                assert_eq!(values.len(), 1);
                assert!(matches!(values[0], Value::Integer(42)));
            }
            _ => panic!("Expected parentheses"),
        }

        let multi_paren = Value::parse_str("(1, 2, 3)").unwrap();
        match multi_paren {
            Value::Paren(ref values) => {
                assert_eq!(values.len(), 3);
                assert!(matches!(values[0], Value::Integer(1)));
                assert!(matches!(values[1], Value::Integer(2)));
                assert!(matches!(values[2], Value::Integer(3)));
            }
            _ => panic!("Expected parentheses with multiple values"),
        }

        let empty_paren = Value::parse_str("()").unwrap();
        match empty_paren {
            Value::Paren(ref values) => {
                assert_eq!(values.len(), 0);
            }
            _ => panic!("Expected empty parentheses"),
        }

        // 测试括号改变优先�?
        let paren_precedence = Value::parse_str("(1 + 2) * 3").unwrap();
        match paren_precedence {
            Value::Binary(BinOp::Mul, left, right) => {
                match *left {
                    Value::Paren(ref values) => {
                        assert_eq!(values.len(), 1);
                        match values[0] {
                            Value::Binary(BinOp::Add, _, _) => (),
                            _ => panic!("Expected addition inside parentheses"),
                        }
                    }
                    _ => panic!("Expected parentheses on left side"),
                }
                assert!(matches!(*right, Value::Integer(3)));
            }
            _ => panic!("Expected multiplication as root"),
        }
    }

    #[test]
    fn test_function_like() {
        // 测试函数调用
        let simple_func = Value::parse_str("add(1, 2)").unwrap();
        match simple_func {
            Value::FuncLike(ref name, ref args) => {
                assert_eq!(name, "add");
                assert_eq!(args.len(), 2);
                assert!(matches!(args[0], Value::Integer(1)));
                assert!(matches!(args[1], Value::Integer(2)));
            }
            _ => panic!("Expected function call"),
        }

        let no_args_func = Value::parse_str("get_value()").unwrap();
        match no_args_func {
            Value::FuncLike(ref name, ref args) => {
                assert_eq!(name, "get_value");
                assert_eq!(args.len(), 0);
            }
            _ => panic!("Expected function call with no arguments"),
        }

        let nested_func = Value::parse_str("outer(inner(5), 10)").unwrap();
        match nested_func {
            Value::FuncLike(ref name, ref args) => {
                assert_eq!(name, "outer");
                assert_eq!(args.len(), 2);
                match args[0] {
                    Value::FuncLike(ref inner_name, ref inner_args) => {
                        assert_eq!(inner_name, "inner");
                        assert_eq!(inner_args.len(), 1);
                        assert!(matches!(inner_args[0], Value::Integer(5)));
                    }
                    _ => panic!("Expected nested function call"),
                }
                assert!(matches!(args[1], Value::Integer(10)));
            }
            _ => panic!("Expected outer function call"),
        }
    }

    #[test]
    fn test_identifiers() {
        // 测试标识�?
        let simple_ident = Value::parse_str("variable").unwrap();
        match simple_ident {
            Value::Ident(ref name) => assert_eq!(name, "variable"),
            _ => panic!("Expected identifier"),
        }

        let underscore_ident = Value::parse_str("_private_var").unwrap();
        match underscore_ident {
            Value::Ident(ref name) => assert_eq!(name, "_private_var"),
            _ => panic!("Expected identifier with underscore"),
        }

        let numeric_ident = Value::parse_str("var123").unwrap();
        match numeric_ident {
            Value::Ident(ref name) => assert_eq!(name, "var123"),
            _ => panic!("Expected identifier with numbers"),
        }
    }

    #[test]
    fn test_complex_expressions() {
        // 测试复杂表达�?
        let complex = Value::parse_str("a = func(b + c, d * e) && !flag").unwrap();
        match complex {
            Value::Assign(AssignOp::Assign, ref ident, ref val) => {
                assert_eq!(ident, "a");
                match val.deref() {
                    Value::Binary(BinOp::And, left, right) => {
                        match left.deref() {
                            Value::FuncLike(ref func_name, ref args) => {
                                assert_eq!(func_name, "func");
                                assert_eq!(args.len(), 2);
                                assert!(matches!(args[0], Value::Binary(BinOp::Add, _, _)));
                                assert!(matches!(args[1], Value::Binary(BinOp::Mul, _, _)));
                            }
                            _ => panic!("Expected function call in assignment"),
                        }
                        match right.deref() {
                            Value::Unary(UnOp::Not, ref val) => {
                                assert!(
                                    matches!(*val.as_ref(), Value::Ident(ref name) if name == "flag")
                                );
                            }
                            _ => panic!("Expected negation on right side"),
                        }
                    }
                    _ => panic!("Expected logical and as root"),
                }
            }
            _ => panic!("Expected assignment on left side"),
        }
    }

    #[test]
    fn test_parse_errors() {
        // 测试解析错误情况
        assert!(Value::parse_str("").is_err());
        assert!(Value::parse_str("+").is_err());
        assert!(Value::parse_str("1 +").is_err());
        assert!(Value::parse_str("(1,)").is_ok()); // 尾随逗号应该是可以的
        assert!(Value::parse_str("func(").is_err());
    }

    #[test]
    #[should_panic]
    fn test_parse_errors2() {
        assert!(Value::parse_str("123abc").is_err()); // 无效的标识符
    }

    #[test]
    fn test_operator_descriptions() {
        // 测试操作符描述方�?
        assert_eq!(UnOp::Not.desc(), "!");
        assert_eq!(UnOp::Neg.desc(), "-");

        assert_eq!(BinOp::Add.desc(), "+");
        assert_eq!(BinOp::Mul.desc(), "*");
        assert_eq!(BinOp::Eq.desc(), "==");
        assert_eq!(BinOp::And.desc(), "&&");
        assert_eq!(BinOp::BitAnd.desc(), "&");

        assert_eq!(AssignOp::Assign.desc(), "=");
        assert_eq!(AssignOp::AddAssign.desc(), "+=");
        assert_eq!(AssignOp::ShlAssign.desc(), "<<=");
    }

    #[test]
    fn test_precedence_ordering() {
        // 测试优先级枚举的排序
        assert!(Precedence::Term > Precedence::Arithmetic);
        assert!(Precedence::Arithmetic > Precedence::Shift);
        assert!(Precedence::Compare > Precedence::Or);
        assert!(Precedence::And > Precedence::Or);
        assert!(Precedence::Any < Precedence::Or);

        // 测试优先级函�?
        assert_eq!(Precedence::of(&BinOp::Mul), Precedence::Term);
        assert_eq!(Precedence::of(&BinOp::Add), Precedence::Arithmetic);
        assert_eq!(Precedence::of(&BinOp::Eq), Precedence::Compare);
        assert_eq!(Precedence::of(&BinOp::And), Precedence::And);
        assert_eq!(Precedence::of(&BinOp::Or), Precedence::Or);
    }
}
