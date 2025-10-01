use super::data::*;

pub trait Valued<T> {
    fn to_i32(&self, ctx: &mut T) -> i32;
}
pub trait ValuedVec<T> {
    fn to_i32_one(&self, ctx: &mut T, idx: usize) -> i32;
    fn to_i32_vec(&self, ctx: &mut T) -> Vec<i32>;
    fn to_i32_last(&self, ctx: &mut T) -> i32;
}
pub trait Context {
    fn call(&mut self, func: &str, values: &Vec<Value>) -> i32;
    fn ident_get(&self, ident: &str) -> i32;
    fn ident_set(&mut self, ident: &str, value: i32);
    fn call_native(&mut self, _func: &str) -> i32 {
        unreachable!()
    }
}
macro_rules! i2b {
    ($expr:expr) => {
        $expr != 0
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
impl UnOp {
    pub fn to_i32<T: Context, V: Valued<T>>(&self, ctx: &mut T, value: V) -> i32 {
        match self {
            UnOp::Not => b2i!(!i2b!(value.to_i32(ctx))),
            UnOp::Neg => -value.to_i32(ctx),
        }
    }
}

impl BinOp {
    pub fn to_i32<T: Context, V: Valued<T>>(&self, ctx: &mut T, left: V, right: V) -> i32 {
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
    pub fn to_i32<T: Context, V: Valued<T>>(&self, ctx: &mut T, ident: &str, value: V) -> i32 {
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
            AssignOp::ShlAssign => ctx.ident_get(ident) << value.to_i32(ctx),
            AssignOp::ShrAssign => ctx.ident_get(ident) >> value.to_i32(ctx),
        };
        ctx.ident_set(ident, v);
        v
    }
}

impl<T: Context> Valued<T> for Value {
    fn to_i32(&self, ctx: &mut T) -> i32 {
        match self {
            Value::Integer(v) => *v,
            Value::Unary(op, v) => op.to_i32(ctx, v),
            Value::Binary(op, l, r) => op.to_i32(ctx, l, r),
            Value::Paren(v) => v.to_i32_last(ctx),
            Value::FuncLike(v, args) => ctx.call(v, args),
            Value::Ident(ident) => ctx.ident_get(ident),
            Value::Assign(op, ident, v) => op.to_i32(ctx, ident, v),
            Value::Native(v) => ctx.call_native(v),
        }
    }
}
impl<T: Context> Valued<T> for &Box<Value> {
    fn to_i32(&self, ctx: &mut T) -> i32 {
        self.as_ref().to_i32(ctx)
    }
}
impl<T: Context> ValuedVec<T> for [Value] {
    fn to_i32_one(&self, ctx: &mut T, idx: usize) -> i32 {
        self.get(idx).map(|e| e.to_i32(ctx)).unwrap_or_default()
    }
    fn to_i32_vec(&self, ctx: &mut T) -> Vec<i32> {
        self.iter().map(|e| e.to_i32(ctx)).collect()
    }
    fn to_i32_last(&self, ctx: &mut T) -> i32 {
        self.iter()
            .map(|e| e.to_i32(ctx))
            .last()
            .unwrap_or_default()
    }
}

pub(crate) use {b2i, i2b};
