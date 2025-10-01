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
impl Value {
    fn as_ident(&self) -> String {
        if let Value::Ident(ident) = self {
            ident.clone()
        } else {
            unreachable!()
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
impl<T> Default for Named<T> {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
        }
    }
}
impl<T: Copy + Default> Named<T> {
    fn get_or_default(&self, key: &str, age: i32) -> T {
        self.get(key, age).copied().unwrap_or_default()
    }
}
impl ContextHelper {
    pub fn call_with<C, H>(ctx: &mut C, helper: &H, func: &str, values: &Vec<Value>) -> i32
    where
        C: Context,
        H: Fn(&mut C) -> &mut ContextHelper,
    {
        ContextHelper1::call(ctx, helper, func, values)
    }
}
impl Context for ContextHelper {
    fn call(&mut self, func: &str, values: &Vec<Value>) -> i32 {
        ContextHelper::call_with(self, &|e| e, func, values)
    }

    fn ident_get(&self, ident: &str) -> i32 {
        self.var_named.get_or_default(ident, self.pointer)
    }

    fn ident_set(&mut self, ident: &str, value: i32) {
        self.var_named.set(ident, value, self.pointer);
    }
}
struct ContextHelper1<C, H>(C, H);
impl<C: Context, H: Fn(&mut C) -> &mut ContextHelper> ContextHelper1<C, H> {
    fn scope_with<F: FnOnce(&mut C) -> i32>(ctx: &mut C, h: &H, func: F) -> i32 {
        let helper = h(ctx);
        let point = helper.pointer;
        helper.pointer += 1;
        let res = func(ctx);
        let helper = h(ctx);
        helper.pointer = point;
        helper.fn_named.clear(point);
        helper.var_named.clear(point);
        res
    }
    fn call_vanilla(ctx: &mut C, h: &H, name: &str, args: Vec<i32>) -> i32 {
        Self::scope_with(ctx, h, |ctx| {
            let helper = h(ctx);
            let pointer = helper.pointer;
            let func = helper
                .fn_named
                .get(name, pointer)
                .map(Arc::clone)
                .expect(&format!("function not found: {}", name));
            for (idx, param) in func.params.iter().enumerate() {
                helper.var_named.set(
                    param,
                    args.get(idx).map(|e| *e).unwrap_or_default(),
                    pointer,
                );
            }
            func.body.to_i32(ctx)
        })
    }
    fn _if(ctx: &mut C, args: &Vec<Value>) -> i32 {
        let res = if i2b!(args.to_i32_one(ctx, 0)) {
            args.to_i32_one(ctx, 1)
        } else {
            args.to_i32_one(ctx, 2)
        };
        if args.len() > 3 {
            args[3..].to_i32_last(ctx);
        }
        res
    }
    fn _while(ctx: &mut C, args: &Vec<Value>) -> i32 {
        let mut res = 0;
        while i2b!(args.to_i32_one(ctx, 0)) {
            res = args[1..].to_i32_last(ctx);
        }
        res
    }
    fn _log(ctx: &mut C, args: &Vec<Value>) -> i32 {
        let msg = args[0].as_ident();
        let args = args[1..].to_i32_vec(ctx);
        println!("{} {:?}", msg, args);
        args.last().map(|e| *e).unwrap_or_default()
    }
    fn _assert(ctx: &mut C, args: &Vec<Value>) -> i32 {
        assert!(i2b!(args.to_i32_one(ctx, 0)));
        args[1..].to_i32_last(ctx)
    }
    fn _fn(ctx: &mut C, h: &H, args: &Vec<Value>) -> i32 {
        let helper = h(ctx);
        helper.fn_named.set(
            &args[0].as_ident(),
            Arc::new(FnDef {
                body: args[1].clone(),
                params: args[2..].iter().map(|e| e.as_ident()).collect(),
            }),
            helper.pointer,
        );
        1
    }
    fn _call(ctx: &mut C, h: &H, args: &Vec<Value>) -> i32 {
        let ident = args[0].as_ident();
        let args = args[1..].to_i32_vec(ctx);
        Self::call_vanilla(ctx, h, &ident, args)
    }
    fn _scope(ctx: &mut C, h: &H, args: &Vec<Value>) -> i32 {
        Self::scope_with(ctx, h, |ctx| args.to_i32_last(ctx))
    }
    fn call(ctx: &mut C, h: &H, func: &str, values: &Vec<Value>) -> i32 {
        match func {
            "_if" => Self::_if(ctx, values),
            "_while" => Self::_while(ctx, values),
            "_log" => Self::_log(ctx, values),
            "_assert" => Self::_assert(ctx, values),
            "_fn" => Self::_fn(ctx, h, values),
            "_call" => Self::_call(ctx, h, values),
            "_scope" => Self::_scope(ctx, h, values),
            _ => {
                let args = values.to_i32_vec(ctx);
                Self::call_vanilla(ctx, h, func, args)
            }
        }
    }
}
