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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unop_debug() {
        // 测试一元操作符的Debug输出
        assert_eq!(format!("{:?}", UnOp::Not), "Not");
        assert_eq!(format!("{:?}", UnOp::Neg), "Neg");
    }

    #[test]
    fn test_binop_debug() {
        // 测试二元操作符的Debug输出
        assert_eq!(format!("{:?}", BinOp::Add), "Add");
        assert_eq!(format!("{:?}", BinOp::Sub), "Sub");
        assert_eq!(format!("{:?}", BinOp::Mul), "Mul");
        assert_eq!(format!("{:?}", BinOp::Div), "Div");
        assert_eq!(format!("{:?}", BinOp::Rem), "Rem");
        assert_eq!(format!("{:?}", BinOp::And), "And");
        assert_eq!(format!("{:?}", BinOp::Or), "Or");
        assert_eq!(format!("{:?}", BinOp::BitXor), "BitXor");
        assert_eq!(format!("{:?}", BinOp::BitAnd), "BitAnd");
        assert_eq!(format!("{:?}", BinOp::BitOr), "BitOr");
        assert_eq!(format!("{:?}", BinOp::Shl), "Shl");
        assert_eq!(format!("{:?}", BinOp::Shr), "Shr");
        assert_eq!(format!("{:?}", BinOp::Eq), "Eq");
        assert_eq!(format!("{:?}", BinOp::Lt), "Lt");
        assert_eq!(format!("{:?}", BinOp::Le), "Le");
        assert_eq!(format!("{:?}", BinOp::Ne), "Ne");
        assert_eq!(format!("{:?}", BinOp::Ge), "Ge");
        assert_eq!(format!("{:?}", BinOp::Gt), "Gt");
    }

    #[test]
    fn test_assignop_debug() {
        // 测试赋值操作符的Debug输出
        assert_eq!(format!("{:?}", AssignOp::Assign), "Assign");
        assert_eq!(format!("{:?}", AssignOp::AddAssign), "AddAssign");
        assert_eq!(format!("{:?}", AssignOp::SubAssign), "SubAssign");
        assert_eq!(format!("{:?}", AssignOp::MulAssign), "MulAssign");
        assert_eq!(format!("{:?}", AssignOp::DivAssign), "DivAssign");
        assert_eq!(format!("{:?}", AssignOp::RemAssign), "RemAssign");
        assert_eq!(format!("{:?}", AssignOp::BitXorAssign), "BitXorAssign");
        assert_eq!(format!("{:?}", AssignOp::BitAndAssign), "BitAndAssign");
        assert_eq!(format!("{:?}", AssignOp::BitOrAssign), "BitOrAssign");
        assert_eq!(format!("{:?}", AssignOp::ShlAssign), "ShlAssign");
        assert_eq!(format!("{:?}", AssignOp::ShrAssign), "ShrAssign");
    }

    #[test]
    fn test_value_variants() {
        // 测试Value枚举的各种变体
        let integer = Value::Integer(42);
        let ident = Value::Ident("test".to_string());
        let native = Value::Native("native_func".to_string());

        match integer {
            Value::Integer(v) => assert_eq!(v, 42),
            _ => panic!("Expected Integer variant"),
        }

        match ident {
            Value::Ident(name) => assert_eq!(name, "test"),
            _ => panic!("Expected Ident variant"),
        }

        match native {
            Value::Native(name) => assert_eq!(name, "native_func"),
            _ => panic!("Expected Native variant"),
        }
    }

    #[test]
    fn test_value_complex_variants() {
        // 测试复杂的Value变体
        let unary = Value::Unary(UnOp::Neg, Box::new(Value::Integer(10)));
        let binary = Value::Binary(
            BinOp::Add,
            Box::new(Value::Integer(1)),
            Box::new(Value::Integer(2)),
        );
        let paren = Value::Paren(vec![Value::Integer(1), Value::Integer(2)]);
        let func_like = Value::FuncLike(
            "add".to_string(),
            vec![Value::Integer(1), Value::Integer(2)],
        );
        let assign = Value::Assign(
            AssignOp::Assign,
            "x".to_string(),
            Box::new(Value::Integer(5)),
        );

        // 验证构造是否正确
        match unary {
            Value::Unary(op, val) => {
                assert!(matches!(op, UnOp::Neg));
                assert!(matches!(val.as_ref(), Value::Integer(10)));
            }
            _ => panic!("Expected Unary variant"),
        }

        match binary {
            Value::Binary(op, left, right) => {
                assert!(matches!(op, BinOp::Add));
                assert!(matches!(left.as_ref(), Value::Integer(1)));
                assert!(matches!(right.as_ref(), Value::Integer(2)));
            }
            _ => panic!("Expected Binary variant"),
        }

        match paren {
            Value::Paren(values) => {
                assert_eq!(values.len(), 2);
                assert!(matches!(values[0], Value::Integer(1)));
                assert!(matches!(values[1], Value::Integer(2)));
            }
            _ => panic!("Expected Paren variant"),
        }

        match func_like {
            Value::FuncLike(name, args) => {
                assert_eq!(name, "add");
                assert_eq!(args.len(), 2);
                assert!(matches!(args[0], Value::Integer(1)));
                assert!(matches!(args[1], Value::Integer(2)));
            }
            _ => panic!("Expected FuncLike variant"),
        }

        match assign {
            Value::Assign(op, ident, val) => {
                assert!(matches!(op, AssignOp::Assign));
                assert_eq!(ident, "x");
                assert!(matches!(val.as_ref(), Value::Integer(5)));
            }
            _ => panic!("Expected Assign variant"),
        }
    }

    #[test]
    fn test_enum_clone() {
        // 测试枚举的克隆功能
        let original = Value::Integer(42);
        let cloned = original.clone();

        match (original, cloned) {
            (Value::Integer(a), Value::Integer(b)) => assert_eq!(a, b),
            _ => panic!("Clone failed"),
        }

        let unop_original = UnOp::Not;
        let unop_cloned = unop_original.clone();
        assert!(matches!(
            (unop_original, unop_cloned),
            (UnOp::Not, UnOp::Not)
        ));

        let binop_original = BinOp::Add;
        let binop_cloned = binop_original.clone();
        assert!(matches!(
            (binop_original, binop_cloned),
            (BinOp::Add, BinOp::Add)
        ));

        let assignop_original = AssignOp::Assign;
        let assignop_cloned = assignop_original.clone();
        assert!(matches!(
            (assignop_original, assignop_cloned),
            (AssignOp::Assign, AssignOp::Assign)
        ));
    }

    #[test]
    fn test_enum_copy() {
        // 测试枚举的拷贝功能（对于实现Copy trait的枚举）
        let unop1 = UnOp::Not;
        let unop2 = unop1; // 这里应该是拷贝，不是移动
        assert!(matches!(unop1, UnOp::Not)); // unop1应该仍然可用
        assert!(matches!(unop2, UnOp::Not));

        let binop1 = BinOp::Add;
        let binop2 = binop1;
        assert!(matches!(binop1, BinOp::Add));
        assert!(matches!(binop2, BinOp::Add));

        let assignop1 = AssignOp::Assign;
        let assignop2 = assignop1;
        assert!(matches!(assignop1, AssignOp::Assign));
        assert!(matches!(assignop2, AssignOp::Assign));
    }
}
