use value_expr::{Context, ContextHelper, Value, Valued};

#[derive(Default)]
struct Context1(ContextHelper);
impl Context for Context1 {
    fn call(&mut self, func: &str, values: &Vec<Value>) -> i32 {
        ContextHelper::call_with(self, &|e| &mut e.0, func, values)
    }

    fn ident_get(&self, ident: &str) -> i32 {
        self.0.ident_get(ident)
    }

    fn ident_set(&mut self, ident: &str, value: i32) {
        self.0.ident_set(ident, value)
    }
}

#[test]
fn demo() {
    let mut ctx = ContextHelper::default();
    let mut ctx1 = Context1::default();
    macro_rules! exec {
        ($expr:literal) => {{
            let v = Value::parse_str($expr).unwrap().to_i32(&mut ctx);
            println!("result: {}", v);
            let v = Value::parse_str($expr).unwrap().to_i32(&mut ctx1);
            println!("result1: {}", v);
        }};
    }

    //条件
    exec!(
        "(
        _assert(_if()==0),
        _assert(_if(1)==0),
        _assert(_if(1,2,3)==2),
        _assert(_if(0,2,3)==3),
        _assert(_if(1,2,3,4)==2),
        _assert(_if(-1>0,2,3)==3),
        )"
    );
    //循环
    exec!(
        "_log(_while,
            i=10,
            _while(i<100000,
                _if(i%10000==0,_log(i_is,i)),
                i+=1,
                i
            )
        )"
    );
    //函数
    exec!(
        "(
            _fn(add,a+b,a,b),
            _assert(add(1)==1),
            _assert(add(1,2)==3),
            _assert(_call(add,1,2)==3)
        )"
    );
    //递归
    exec!(
        "(
            _fn(fib1,_if(n<2,a2,fib1(n-1,a2,a1+a2)),n,a1,a2),
            _fn(fib,fib1(n,1,1),n),
            _log(fib,fib(0),fib(1),fib(2),fib(3),fib(10),fib(19)),
            _assert(6765==fib(19))
        )"
    );
    //作用域
    exec!(
        "(
            _scope(a=100,_log(a,a),_assert(a==100)),
            _scope(a=100,_scope(_assert(a==100))),
            _scope(a=100,a=200,_assert(a==200)),
            _scope(a=100,_scope(a=200),_assert(a==100)),
            _fn(f1,_assert(a==0)),
            _scope(a=100,_fn(f1,_assert(a==100))),
            _scope(a=100,_fn(f1,(a=200,_assert(a==200))),_assert(a==100))
        )"
    );

    //nop
    drop(ctx);
    drop(ctx1);
}
