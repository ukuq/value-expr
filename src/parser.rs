use super::data::*;
use proc_macro2::TokenStream;
use std::str::FromStr;
use syn::ext::IdentExt;
use syn::parse::discouraged::Speculative;
use syn::parse::{Parse, ParseStream};
use syn::{parenthesized, token, Ident, LitInt, Token};

macro_rules! op {
    ($name:ident => $msg:literal; $($tt:tt => $ident:ident);+$(;)?) => {
        impl Parse for $name {
            fn parse(input: ParseStream) -> syn::Result<Self> {
                let lookahead = input.lookahead1();
                $(if lookahead.peek(Token!$tt) {
                    input.parse::<Token!$tt>().map(|_|Self::$ident)
                } else)+ {
                    Err(input.error($msg))
                }
            }
        }
        impl $name {
            #[allow(unused)]
            pub fn desc(self) -> &'static str {
                match self {
                    $(Self::$ident => stringify!($tt)),+
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
