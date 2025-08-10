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
            let func = helper.fn_named.get(name, pointer).unwrap().clone();
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
}
