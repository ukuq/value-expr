use super::data::*;

pub trait Valued<T: ?Sized> {
    fn to_i32(&self, ctx: &mut T) -> i32;
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

impl<T: Context, V: Valued<T>> Valued<T> for [V] {
    fn to_i32(&self, ctx: &mut T) -> i32 {
        let mut last = 0;
        for value in self.iter() {
            last = value.to_i32(ctx);
        }
        last
    }
}

impl<T: Context> Valued<T> for Value {
    fn to_i32(&self, ctx: &mut T) -> i32 {
        match self {
            Value::Integer(v) => *v,
            Value::Unary(op, v) => op.to_i32(ctx, v),
            Value::Binary(op, l, r) => op.to_i32(ctx, l, r),
            Value::Paren(v) => v.to_i32(ctx),
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

pub(crate) use {b2i, i2b};
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // 测试上下文实现
    struct TestContext {
        variables: HashMap<String, i32>,
        functions: HashMap<String, Box<dyn Fn(&[i32]) -> i32>>,
    }

    impl Default for TestContext {
        fn default() -> Self {
            let mut functions: HashMap<String, Box<dyn Fn(&[i32]) -> i32>> = HashMap::new();
            functions.insert(
                "add".to_string(),
                Box::new(|args: &[i32]| args.iter().sum()),
            );
            functions.insert(
                "mul".to_string(),
                Box::new(|args: &[i32]| args.iter().product()),
            );
            functions.insert(
                "max".to_string(),
                Box::new(|args: &[i32]| *args.iter().max().unwrap_or(&0)),
            );
            functions.insert(
                "min".to_string(),
                Box::new(|args: &[i32]| *args.iter().min().unwrap_or(&0)),
            );

            Self {
                variables: HashMap::new(),
                functions,
            }
        }
    }

    impl Context for TestContext {
        fn call(&mut self, func: &str, values: &Vec<Value>) -> i32 {
            let args: Vec<_> = values.iter().map(|e| e.to_i32(self)).collect();
            if let Some(f) = self.functions.get(func) {
                f(&args)
            } else {
                panic!("Unknown function: {}", func)
            }
        }

        fn ident_get(&self, ident: &str) -> i32 {
            self.variables.get(ident).copied().unwrap_or(0)
        }

        fn ident_set(&mut self, ident: &str, value: i32) {
            self.variables.insert(ident.to_string(), value);
        }
    }

    impl TestContext {
        fn eval(&mut self, str: &str) -> i32 {
            Value::parse_str(str).unwrap().to_i32(self)
        }
    }

    #[test]
    fn test_basic_operations() {
        let mut ctx = TestContext::default();

        // 基本算术
        assert_eq!(ctx.eval("1+1"), 2);
        assert_eq!(ctx.eval("5-3"), 2);
        assert_eq!(ctx.eval("3*4"), 12);
        assert_eq!(ctx.eval("8/2"), 4);
        assert_eq!(ctx.eval("7%3"), 1);

        // 函数调用
        assert_eq!(ctx.eval("add(1,2,3)"), 6);
        assert_eq!(ctx.eval("mul(2,3,4)"), 24);
        assert_eq!(ctx.eval("max(1,5,3)"), 5);
        assert_eq!(ctx.eval("min(7,2,9)"), 2);
    }

    #[test]
    fn test_unary_operators() {
        let mut ctx = TestContext::default();

        // 测试一元负号
        assert_eq!(UnOp::Neg.to_i32(&mut ctx, Value::Integer(5)), -5);
        assert_eq!(UnOp::Neg.to_i32(&mut ctx, Value::Integer(-3)), 3);
        assert_eq!(UnOp::Neg.to_i32(&mut ctx, Value::Integer(0)), 0);

        // 测试一元非操作
        assert_eq!(UnOp::Not.to_i32(&mut ctx, Value::Integer(1)), 0);
        assert_eq!(UnOp::Not.to_i32(&mut ctx, Value::Integer(0)), 1);
        assert_eq!(UnOp::Not.to_i32(&mut ctx, Value::Integer(42)), 0);
        assert_eq!(UnOp::Not.to_i32(&mut ctx, Value::Integer(-1)), 0);

        // 通过表达式测试
        assert_eq!(ctx.eval("-5"), -5);
        assert_eq!(ctx.eval("!0"), 1);
        assert_eq!(ctx.eval("!1"), 0);
        assert_eq!(ctx.eval("!!1"), 1);
    }

    #[test]
    fn test_binary_arithmetic_operators() {
        let mut ctx = TestContext::default();

        // 加法
        assert_eq!(
            BinOp::Add.to_i32(&mut ctx, Value::Integer(3), Value::Integer(5)),
            8
        );
        assert_eq!(
            BinOp::Add.to_i32(&mut ctx, Value::Integer(-2), Value::Integer(7)),
            5
        );

        // 减法
        assert_eq!(
            BinOp::Sub.to_i32(&mut ctx, Value::Integer(10), Value::Integer(3)),
            7
        );
        assert_eq!(
            BinOp::Sub.to_i32(&mut ctx, Value::Integer(5), Value::Integer(8)),
            -3
        );

        // 乘法
        assert_eq!(
            BinOp::Mul.to_i32(&mut ctx, Value::Integer(4), Value::Integer(6)),
            24
        );
        assert_eq!(
            BinOp::Mul.to_i32(&mut ctx, Value::Integer(-3), Value::Integer(5)),
            -15
        );

        // 除法
        assert_eq!(
            BinOp::Div.to_i32(&mut ctx, Value::Integer(15), Value::Integer(3)),
            5
        );
        assert_eq!(
            BinOp::Div.to_i32(&mut ctx, Value::Integer(7), Value::Integer(2)),
            3
        );

        // 取模
        assert_eq!(
            BinOp::Rem.to_i32(&mut ctx, Value::Integer(10), Value::Integer(3)),
            1
        );
        assert_eq!(
            BinOp::Rem.to_i32(&mut ctx, Value::Integer(8), Value::Integer(4)),
            0
        );
    }

    #[test]
    fn test_binary_bitwise_operators() {
        let mut ctx = TestContext::default();

        // 按位与
        assert_eq!(
            BinOp::BitAnd.to_i32(&mut ctx, Value::Integer(0b1010), Value::Integer(0b1100)),
            0b1000
        );
        assert_eq!(
            BinOp::BitAnd.to_i32(&mut ctx, Value::Integer(15), Value::Integer(7)),
            7
        );

        // 按位或
        assert_eq!(
            BinOp::BitOr.to_i32(&mut ctx, Value::Integer(0b1010), Value::Integer(0b1100)),
            0b1110
        );
        assert_eq!(
            BinOp::BitOr.to_i32(&mut ctx, Value::Integer(8), Value::Integer(4)),
            12
        );

        // 按位异或
        assert_eq!(
            BinOp::BitXor.to_i32(&mut ctx, Value::Integer(0b1010), Value::Integer(0b1100)),
            0b0110
        );
        assert_eq!(
            BinOp::BitXor.to_i32(&mut ctx, Value::Integer(5), Value::Integer(3)),
            6
        );

        // 左移
        assert_eq!(
            BinOp::Shl.to_i32(&mut ctx, Value::Integer(5), Value::Integer(2)),
            20
        );
        assert_eq!(
            BinOp::Shl.to_i32(&mut ctx, Value::Integer(1), Value::Integer(3)),
            8
        );

        // 右移
        assert_eq!(
            BinOp::Shr.to_i32(&mut ctx, Value::Integer(20), Value::Integer(2)),
            5
        );
        assert_eq!(
            BinOp::Shr.to_i32(&mut ctx, Value::Integer(8), Value::Integer(3)),
            1
        );
    }

    #[test]
    fn test_binary_logical_operators() {
        let mut ctx = TestContext::default();

        // 逻辑与
        assert_eq!(
            BinOp::And.to_i32(&mut ctx, Value::Integer(1), Value::Integer(1)),
            1
        );
        assert_eq!(
            BinOp::And.to_i32(&mut ctx, Value::Integer(1), Value::Integer(0)),
            0
        );
        assert_eq!(
            BinOp::And.to_i32(&mut ctx, Value::Integer(0), Value::Integer(1)),
            0
        );
        assert_eq!(
            BinOp::And.to_i32(&mut ctx, Value::Integer(0), Value::Integer(0)),
            0
        );
        assert_eq!(
            BinOp::And.to_i32(&mut ctx, Value::Integer(5), Value::Integer(3)),
            1
        );

        // 逻辑或
        assert_eq!(
            BinOp::Or.to_i32(&mut ctx, Value::Integer(1), Value::Integer(1)),
            1
        );
        assert_eq!(
            BinOp::Or.to_i32(&mut ctx, Value::Integer(1), Value::Integer(0)),
            1
        );
        assert_eq!(
            BinOp::Or.to_i32(&mut ctx, Value::Integer(0), Value::Integer(1)),
            1
        );
        assert_eq!(
            BinOp::Or.to_i32(&mut ctx, Value::Integer(0), Value::Integer(0)),
            0
        );
        assert_eq!(
            BinOp::Or.to_i32(&mut ctx, Value::Integer(-1), Value::Integer(0)),
            1
        );
    }

    #[test]
    fn test_binary_comparison_operators() {
        let mut ctx = TestContext::default();

        // 等于
        assert_eq!(
            BinOp::Eq.to_i32(&mut ctx, Value::Integer(5), Value::Integer(5)),
            1
        );
        assert_eq!(
            BinOp::Eq.to_i32(&mut ctx, Value::Integer(5), Value::Integer(3)),
            0
        );

        // 不等于
        assert_eq!(
            BinOp::Ne.to_i32(&mut ctx, Value::Integer(5), Value::Integer(3)),
            1
        );
        assert_eq!(
            BinOp::Ne.to_i32(&mut ctx, Value::Integer(5), Value::Integer(5)),
            0
        );

        // 小于
        assert_eq!(
            BinOp::Lt.to_i32(&mut ctx, Value::Integer(3), Value::Integer(5)),
            1
        );
        assert_eq!(
            BinOp::Lt.to_i32(&mut ctx, Value::Integer(5), Value::Integer(3)),
            0
        );
        assert_eq!(
            BinOp::Lt.to_i32(&mut ctx, Value::Integer(5), Value::Integer(5)),
            0
        );

        // 小于等于
        assert_eq!(
            BinOp::Le.to_i32(&mut ctx, Value::Integer(3), Value::Integer(5)),
            1
        );
        assert_eq!(
            BinOp::Le.to_i32(&mut ctx, Value::Integer(5), Value::Integer(5)),
            1
        );
        assert_eq!(
            BinOp::Le.to_i32(&mut ctx, Value::Integer(5), Value::Integer(3)),
            0
        );

        // 大于
        assert_eq!(
            BinOp::Gt.to_i32(&mut ctx, Value::Integer(5), Value::Integer(3)),
            1
        );
        assert_eq!(
            BinOp::Gt.to_i32(&mut ctx, Value::Integer(3), Value::Integer(5)),
            0
        );
        assert_eq!(
            BinOp::Gt.to_i32(&mut ctx, Value::Integer(5), Value::Integer(5)),
            0
        );

        // 大于等于
        assert_eq!(
            BinOp::Ge.to_i32(&mut ctx, Value::Integer(5), Value::Integer(3)),
            1
        );
        assert_eq!(
            BinOp::Ge.to_i32(&mut ctx, Value::Integer(5), Value::Integer(5)),
            1
        );
        assert_eq!(
            BinOp::Ge.to_i32(&mut ctx, Value::Integer(3), Value::Integer(5)),
            0
        );
    }

    #[test]
    fn test_assignment_operators() {
        let mut ctx = TestContext::default();

        // 简单赋值
        assert_eq!(
            AssignOp::Assign.to_i32(&mut ctx, "x", Value::Integer(10)),
            10
        );
        assert_eq!(ctx.ident_get("x"), 10);

        // 加法赋值
        ctx.ident_set("y", 5);
        assert_eq!(
            AssignOp::AddAssign.to_i32(&mut ctx, "y", Value::Integer(3)),
            8
        );
        assert_eq!(ctx.ident_get("y"), 8);

        // 减法赋值
        ctx.ident_set("z", 10);
        assert_eq!(
            AssignOp::SubAssign.to_i32(&mut ctx, "z", Value::Integer(4)),
            6
        );
        assert_eq!(ctx.ident_get("z"), 6);

        // 乘法赋值
        ctx.ident_set("a", 3);
        assert_eq!(
            AssignOp::MulAssign.to_i32(&mut ctx, "a", Value::Integer(4)),
            12
        );
        assert_eq!(ctx.ident_get("a"), 12);

        // 除法赋值
        ctx.ident_set("b", 20);
        assert_eq!(
            AssignOp::DivAssign.to_i32(&mut ctx, "b", Value::Integer(4)),
            5
        );
        assert_eq!(ctx.ident_get("b"), 5);

        // 取模赋值
        ctx.ident_set("c", 13);
        assert_eq!(
            AssignOp::RemAssign.to_i32(&mut ctx, "c", Value::Integer(5)),
            3
        );
        assert_eq!(ctx.ident_get("c"), 3);
    }

    #[test]
    fn test_macro_functionality() {
        // 测试i2b宏
        assert_eq!(i2b!(5), true);
        assert_eq!(i2b!(0), false);
        assert_eq!(i2b!(-1), true);

        // 测试b2i宏
        assert_eq!(b2i!(true), 1);
        assert_eq!(b2i!(false), 0);
        assert_eq!(b2i!(5 > 3), 1);
        assert_eq!(b2i!(2 < 1), 0);
    }
}
