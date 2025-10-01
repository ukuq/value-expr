#[derive(Debug, Copy, Clone)]
pub enum UnOp {
    Not,
    Neg,
}
#[derive(Debug, Copy, Clone)]
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
#[derive(Debug, Copy, Clone)]
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
    ShlAssign,
    ShrAssign,
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
    /// native use
    #[allow(unused)]
    Native(String),
}
