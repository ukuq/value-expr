use super::data::*;
use super::valuer::*;
use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

#[derive(Debug)]
struct Aged<T> {
    age: i32,
    value: T,
}
#[derive(Debug)]
struct Named<T> {
    map: HashMap<String, Vec<Aged<T>>>,
}
#[derive(Debug)]
struct FnDef {
    body: Value,
    params: Vec<String>,
}
#[derive(Default)]
pub struct ContextHelper {
    pointer: i32,
    fn_named: Named<Arc<FnDef>>,
    var_named: Named<i32>,
}
pub trait ContextHolder: Sized {
    fn ctx(&mut self) -> &mut ContextHelper;
    fn ctx_ref(&self) -> &ContextHelper;
    fn ctx_log(&self, msg: &str) {
        println!("{}", msg);
    }
    fn ctx_call(&mut self, _func: &str, _values: &Vec<Value>) -> Option<i32> {
        None
    }
}
impl<T> Default for Named<T> {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
        }
    }
}
impl<T> Named<T> {
    fn get(&self, key: &str, age: i32) -> Option<&T> {
        match self.map.get(key) {
            None => None,
            Some(vec) => vec.iter().rev().find(|a| a.age <= age).map(|a| &a.value),
        }
    }
    fn set(&mut self, key: &str, value: T, age: i32) -> Option<T> {
        let frames = self.map.entry(key.to_string()).or_insert(Vec::new());
        while let Some(frame) = frames.last_mut() {
            if frame.age > age {
                frames.pop();
            } else if frame.age == age {
                return Some(mem::replace(&mut frame.value, value));
            } else {
                break;
            }
        }
        frames.push(Aged { age, value });
        None
    }
    fn clear(&mut self, age: i32) {
        let mut need_remove = vec![];
        for (key, frames) in &mut self.map {
            while let Some(frame) = frames.last_mut() {
                if frame.age > age {
                    frames.pop();
                } else {
                    break;
                }
            }
            if frames.is_empty() {
                need_remove.push(key.to_string());
            }
        }
        for key in need_remove {
            self.map.remove(&key);
        }
    }
}
impl ContextHelper {
    fn ident_get(&self, ident: &str) -> i32 {
        let age = self.pointer;
        self.var_named
            .get(ident, age)
            .map(|e| *e)
            .unwrap_or_default()
    }
    fn ident_set(&mut self, ident: &str, value: i32) {
        let age = self.pointer;
        self.var_named.set(ident, value, age);
    }
    fn fn_def(&mut self, name: &str, body: Value, params: Vec<String>) {
        self.fn_named
            .set(name, Arc::new(FnDef { body, params }), self.pointer);
    }
}
trait ContextHelper0<T: ContextHolder> {
    fn scope_with<F: FnOnce(&mut T) -> i32>(ctx: &mut T, func: F) -> i32 {
        let helper = ctx.ctx();
        let point = helper.pointer;
        helper.pointer += 1;
        let res = func(ctx);
        let helper = ctx.ctx();
        helper.pointer = point;
        helper.fn_named.clear(point);
        helper.var_named.clear(point);
        res
    }
    fn fn_call(ctx: &mut T, name: &str, args: Vec<i32>) -> i32 {
        Self::scope_with(ctx, |ctx| {
            let helper = ctx.ctx();
            let pointer = helper.pointer;
            let func = helper
                .fn_named
                .get(name, pointer)
                .expect(&format!("function not exist: {}", name))
                .clone();
            for (idx, param) in func.params.iter().enumerate() {
                helper.var_named.set(
                    param,
                    args.get(idx).map(|e| *e).unwrap_or_default(),
                    pointer,
                );
            }
            let res = func.body.to_i32(ctx);
            res
        })
    }
    fn _if(ctx: &mut T, args: &Vec<Value>) -> i32 {
        match args.len() {
            0 => 0,
            1 => {
                if i2b!(args[0].to_i32(ctx)) {
                    0
                } else {
                    0
                }
            }
            2 => {
                if i2b!(args[0].to_i32(ctx)) {
                    args[1].to_i32(ctx)
                } else {
                    0
                }
            }
            _ => {
                let res = if i2b!(args[0].to_i32(ctx)) {
                    args[1].to_i32(ctx)
                } else {
                    args[2].to_i32(ctx)
                };
                args[3..].to_i32(ctx);
                res
            }
        }
    }
    fn _while(ctx: &mut T, args: &Vec<Value>) -> i32 {
        let mut res = 0;
        while i2b!(args[0].to_i32(ctx)) {
            res = args[1..].to_i32(ctx);
        }
        res
    }
    fn _log(ctx: &mut T, args: &Vec<Value>) -> i32 {
        let msg = args[0].as_ident();
        let args = args[1..].to_i32_vec(ctx);
        ctx.ctx_log(&format!("{} {:?}", msg, args));
        args.last().map(|e| *e).unwrap_or(0)
    }
    fn _assert(ctx: &mut T, args: &Vec<Value>) -> i32 {
        assert!(i2b!(args[0].to_i32(ctx)));
        1
    }
    fn _fn(ctx: &mut T, args: &Vec<Value>) -> i32 {
        ctx.ctx().fn_def(
            &args[0].as_ident(),
            args[1].clone(),
            args[2..].iter().map(|e| e.as_ident()).collect(),
        );
        1
    }
    fn _call(ctx: &mut T, args: &Vec<Value>) -> i32 {
        let ident = args[0].as_ident();
        let args = args[1..].to_i32_vec(ctx);
        Self::fn_call(ctx, &ident, args)
    }
    fn _scope(ctx: &mut T, args: &Vec<Value>) -> i32 {
        Self::scope_with(ctx, |ctx| args.to_i32(ctx))
    }
    fn call_with(ctx: &mut T, func: &str, values: &Vec<Value>) -> i32 {
        match func {
            "_if" => Self::_if(ctx, values),
            "_while" => Self::_while(ctx, values),
            "_log" => Self::_log(ctx, values),
            "_assert" => Self::_assert(ctx, values),
            "_fn" => Self::_fn(ctx, values),
            "_call" => Self::_call(ctx, values),
            "_scope" => Self::_scope(ctx, values),
            _ => {
                let args = values.to_i32_vec(ctx);
                Self::fn_call(ctx, func, args)
            }
        }
    }
}
impl<T: ContextHolder> ContextHelper0<T> for ContextHelper {}

impl Value {
    fn as_ident(&self) -> String {
        if let Value::Ident(ident) = self {
            ident.clone()
        } else {
            unreachable!()
        }
    }
}

trait ValuedVec<T> {
    fn to_i32_vec(&self, ctx: &mut T) -> Vec<i32>;
}
impl<T: Context, V: Valued<T>> ValuedVec<T> for [V] {
    fn to_i32_vec(&self, ctx: &mut T) -> Vec<i32> {
        self.iter().map(|e| e.to_i32(ctx)).collect()
    }
}

impl<T: ContextHolder> Context for T {
    fn call(&mut self, func: &str, values: &Vec<Value>) -> i32 {
        match self.ctx_call(func, values) {
            Some(res) => res,
            None => ContextHelper::call_with(self, func, values),
        }
    }

    fn ident_get(&self, ident: &str) -> i32 {
        self.ctx_ref().ident_get(ident)
    }

    fn ident_set(&mut self, ident: &str, value: i32) {
        self.ctx().ident_set(ident, value)
    }
}

impl ContextHolder for ContextHelper {
    fn ctx(&mut self) -> &mut ContextHelper {
        self
    }

    fn ctx_ref(&self) -> &ContextHelper {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl ContextHelper {
        pub fn exec(&mut self, str: &str) {
            let v = Value::parse_str(str).unwrap().to_i32(self);
            println!("exec_value_is: {}", v);
        }
    }

    #[test]
    fn test() {
        let mut ctx = ContextHelper::default();

        //条件
        ctx.exec(
            "(
            _assert(_if(1,2,3)==2),
            _assert(_if(-1>0,2,3)==3),
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
            _scope(a=100,_log(a,a),_assert(a==100)),
            _scope(a=100,_scope(_assert(a==100))),
            _scope(a=100,a=200,_assert(a==200)),
            _scope(a=100,_scope(a=200),_assert(a==100)),
            _fn(f1,_assert(a==0)),
            _scope(a=100,_fn(f1,_assert(a==100))),
            _scope(a=100,_fn(f1,(a=200,_assert(a==200))),_assert(a==100))
            )",
        );
    }

    #[test]
    fn test_named_operations() {
        // 测试Named结构的基本操作
        let mut named: Named<i32> = Named::default();

        // 测试设置和获取
        assert_eq!(named.set("test", 42, 0), None);
        assert_eq!(named.get("test", 0), Some(&42));
        assert_eq!(named.get("test", 1), Some(&42)); // 更高的age应该也能访问

        // 测试覆盖相同age的值
        assert_eq!(named.set("test", 84, 0), Some(42));
        assert_eq!(named.get("test", 0), Some(&84));

        // 测试不同age的值
        named.set("test", 100, 1);
        assert_eq!(named.get("test", 0), Some(&84)); // age 0 应该看到 age 0 的值
        assert_eq!(named.get("test", 1), Some(&100)); // age 1 应该看到 age 1 的值

        // 测试获取不存在的键
        assert_eq!(named.get("nonexistent", 0), None);
    }

    #[test]
    fn test_named_clear() {
        let mut named: Named<i32> = Named::default();

        // 设置不同age的值
        named.set("a", 10, 0);
        named.set("a", 20, 1);
        named.set("a", 30, 2);
        named.set("b", 40, 1);

        // 清理age > 1的值
        named.clear(1);

        // 验证清理结果
        assert_eq!(named.get("a", 0), Some(&10));
        assert_eq!(named.get("a", 1), Some(&20));
        assert_eq!(named.get("a", 2), Some(&20)); // 应该回退到age 1的值
        assert_eq!(named.get("b", 1), Some(&40));

        // 清理age > 0的值
        named.clear(0);

        assert_eq!(named.get("a", 0), Some(&10));
        assert_eq!(named.get("a", 1), Some(&10)); // 应该回退到age 0的值
        assert_eq!(named.get("b", 1), None); // b应该被完全清除
    }

    #[test]
    fn test_context_helper_variables() {
        let mut helper = ContextHelper::default();

        // 测试变量的设置和获取
        helper.ident_set("x", 42);
        assert_eq!(helper.ident_get("x"), 42);

        // 测试不存在的变量返回默认值
        assert_eq!(helper.ident_get("nonexistent"), 0);

        // 测试变量覆盖
        helper.ident_set("x", 84);
        assert_eq!(helper.ident_get("x"), 84);
    }

    #[test]
    fn test_context_helper_functions() {
        let mut helper = ContextHelper::default();

        // 定义一个简单函数
        let body = Value::Binary(
            crate::BinOp::Add,
            Box::new(Value::Ident("a".to_string())),
            Box::new(Value::Ident("b".to_string())),
        );
        helper.fn_def("add", body, vec!["a".to_string(), "b".to_string()]);

        // 通过内部机制测试函数是否被正确存储
        assert!(helper.fn_named.get("add", 0).is_some());
    }

    #[test]
    fn test_built_in_functions() {
        let mut ctx = ContextHelper::default();

        // 测试_if函数的各种情况
        let if_args_0 = vec![];
        assert_eq!(ContextHelper::_if(&mut ctx, &if_args_0), 0);

        let if_args_1 = vec![Value::Integer(1)];
        assert_eq!(ContextHelper::_if(&mut ctx, &if_args_1), 0);

        let if_args_2 = vec![Value::Integer(1), Value::Integer(42)];
        assert_eq!(ContextHelper::_if(&mut ctx, &if_args_2), 42);

        let if_args_2_false = vec![Value::Integer(0), Value::Integer(42)];
        assert_eq!(ContextHelper::_if(&mut ctx, &if_args_2_false), 0);

        let if_args_3 = vec![Value::Integer(1), Value::Integer(42), Value::Integer(84)];
        assert_eq!(ContextHelper::_if(&mut ctx, &if_args_3), 42);

        let if_args_3_false = vec![Value::Integer(0), Value::Integer(42), Value::Integer(84)];
        assert_eq!(ContextHelper::_if(&mut ctx, &if_args_3_false), 84);

        // 测试_assert函数
        let assert_args_true = vec![Value::Integer(1)];
        assert_eq!(ContextHelper::_assert(&mut ctx, &assert_args_true), 1);

        // 测试_scope函数
        let scope_args = vec![
            Value::Assign(
                crate::AssignOp::Assign,
                "x".to_string(),
                Box::new(Value::Integer(100)),
            ),
            Value::Ident("x".to_string()),
        ];
        let result = ContextHelper::_scope(&mut ctx, &scope_args);
        assert_eq!(result, 100);
        // 验证作用域外变量不存在
        assert_eq!(ctx.ident_get("x"), 0);
    }

    #[test]
    fn test_context_holder_trait() {
        let mut helper = ContextHelper::default();

        // 测试ctx方法
        let ctx_ref = helper.ctx();
        assert_eq!(ctx_ref.pointer, 0);

        // 测试ctx_ref方法
        let ctx_ref = helper.ctx_ref();
        assert_eq!(ctx_ref.pointer, 0);

        // 测试ctx_call默认实现
        let values = vec![Value::Integer(1)];
        assert_eq!(helper.ctx_call("test", &values), None);
    }

    #[test]
    fn test_value_as_ident() {
        let ident_value = Value::Ident("test_var".to_string());
        assert_eq!(ident_value.as_ident(), "test_var");

        let another_ident = Value::Ident("another".to_string());
        assert_eq!(another_ident.as_ident(), "another");
    }

    #[test]
    fn test_context_integration() {
        let mut ctx = ContextHelper::default();

        // 测试标识符获取和设置
        assert_eq!(ctx.ident_get("undefined"), 0);
        ctx.ident_set("test", 42);
        assert_eq!(ctx.ident_get("test"), 42);

        // 测试函数调用机制
        //let add_args = vec![Value::Integer(3), Value::Integer(5)];
        // 由于没有定义add函数，这会触发错误处理
    }

    #[test]
    fn test_aged_structure() {
        // 测试Aged结构的基本功能
        let aged = Aged {
            age: 5,
            value: "test".to_string(),
        };
        assert_eq!(aged.age, 5);
        assert_eq!(aged.value, "test");
    }

    #[test]
    fn test_fn_def_structure() {
        // 测试FnDef结构
        let body = Value::Integer(42);
        let params = vec!["a".to_string(), "b".to_string()];
        let fn_def = FnDef { body, params };

        assert!(matches!(fn_def.body, Value::Integer(42)));
        assert_eq!(fn_def.params.len(), 2);
        assert_eq!(fn_def.params[0], "a");
        assert_eq!(fn_def.params[1], "b");
    }

    #[test]
    fn test_context_helper_defaults() {
        let helper = ContextHelper::default();

        // 测试默认值
        assert_eq!(helper.pointer, 0);
        assert!(helper.fn_named.map.is_empty());
        assert!(helper.var_named.map.is_empty());
    }

    #[test]
    fn test_log_function() {
        let mut ctx = ContextHelper::default();

        // 测试_log函数
        let log_args = vec![
            Value::Ident("test_message".to_string()),
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ];
        let result = ContextHelper::_log(&mut ctx, &log_args);
        assert_eq!(result, 3); // 应该返回最后一个参数的值

        // 测试只有消息没有参数的情况
        let log_args_minimal = vec![Value::Ident("message".to_string())];
        let result = ContextHelper::_log(&mut ctx, &log_args_minimal);
        assert_eq!(result, 0); // 没有参数时应该返回0
    }

    #[test]
    fn test_while_function() {
        let mut ctx = ContextHelper::default();

        // 测试_while函数的基本功能
        // 设置一个计数器变量
        ctx.ident_set("counter", 0);

        // 创建一个简单的while循环：while(counter < 3, counter = counter + 1)
        let while_args = vec![
            Value::Binary(
                crate::BinOp::Lt,
                Box::new(Value::Ident("counter".to_string())),
                Box::new(Value::Integer(3)),
            ),
            Value::Assign(
                crate::AssignOp::Assign,
                "counter".to_string(),
                Box::new(Value::Binary(
                    crate::BinOp::Add,
                    Box::new(Value::Ident("counter".to_string())),
                    Box::new(Value::Integer(1)),
                )),
            ),
        ];

        let result = ContextHelper::_while(&mut ctx, &while_args);
        assert_eq!(ctx.ident_get("counter"), 3);
        assert_eq!(result, 3); // 应该返回最后一次执行的结果
    }

    #[test]
    fn test_complex_scope_operations() {
        let mut ctx = ContextHelper::default();

        // 设置全局变量
        ctx.ident_set("global", 10);

        // 测试嵌套作用域
        let result = ContextHelper::scope_with(&mut ctx, |ctx| {
            // 在内部作用域设置局部变量
            ctx.ident_set("local", 20);
            assert_eq!(ctx.ident_get("global"), 10); // 能访问外部变量
            assert_eq!(ctx.ident_get("local"), 20); // 能访问局部变量

            // 再次嵌套作用域
            ContextHelper::scope_with(ctx, |ctx| {
                ctx.ident_set("inner", 30);
                assert_eq!(ctx.ident_get("global"), 10);
                assert_eq!(ctx.ident_get("local"), 20);
                assert_eq!(ctx.ident_get("inner"), 30);
                42 // 返回值
            })
        });

        assert_eq!(result, 42);
        assert_eq!(ctx.ident_get("global"), 10); // 全局变量应该保持
        assert_eq!(ctx.ident_get("local"), 0); // 局部变量应该被清理
        assert_eq!(ctx.ident_get("inner"), 0); // 内部变量应该被清理
    }
}
