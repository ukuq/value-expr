use value_expr::{ContextHelper, Value, Valued};

#[test]
fn demo() {
    let mut ctx = ContextHelper::default();
    macro_rules! exec {
        ($expr:literal) => {{
            let v = Value::parse_str($expr).unwrap().to_i32(&mut ctx);
            println!("exec_value_is: {}", v);
        }};
    }

    //条件
    exec!(
        "(
            _assert(_if(1,2,3)==2),
            _assert(_if(-1>0,2,3)==3),
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
}
