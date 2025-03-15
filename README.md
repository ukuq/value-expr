rust-expr
=========

Easy expression.

```rust
fn demo() {
    let mut ctx = DemoContext::default();
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
            _scope(a=100,_assert(a==100)),
            _scope(a=100,_scope(_assert(a==100))),
            _scope(a=100,a=200,_assert(a==200)),
            _scope(a=100,_scope(a=200),_assert(a==100)),
            _fn(f1,_assert(a==0)),
            _scope(a=100,_fn(f1,_assert(a==100))),
            _scope(a=100,_fn(f1,(a=200,_assert(a==200))),_assert(a==100))
            )",
    );
}
```