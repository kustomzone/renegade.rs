pub struct Renegade {

} 

pub trait MetricExtractor<I> {
    fn distance(input_a : I, input_b : I) -> f64;
}

pub trait Row<I, O> {
    fn input() -> I;
    fn output() -> O;
}

impl Renegade {
    pub fn new<I, O, TI, T>(trainingData : T) -> Renegade 
    where 
     TI : Row<I, O>,
     T : IntoIterator<Item = TI>
    {
        Renegade {}
    }
}