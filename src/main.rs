use std::time::SystemTime;
use value_expr::{IValue, Value};

fn main() {
    test(|| {
        let v = Value::parse_str("1+2*3").unwrap();
        v.to_i32(&mut ());
    });
}

fn test(f: impl Fn()) {
    let start = SystemTime::now();
    f();
    let end = SystemTime::now();
    let nanos = end.duration_since(start).unwrap().subsec_nanos();
    println!("time nanos: {}", nanos);
}
