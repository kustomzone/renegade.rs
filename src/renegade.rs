pub struct Renegade {

} 

pub trait Labelled {
    fn label(&self) -> &str;
}

pub trait Metric<InputType> {
    fn distance(&self, input_a : InputType, input_b : InputType) -> f64;
}

pub trait Row<InputType, OutputType> {
    fn input(&self) -> InputType;
    fn output(&self) -> OutputType;
}

impl Renegade {
    pub fn new<InputType, OutputType, InstanceType, TrainingDataType, MetricType>(
        training_data : TrainingDataType, 
        input_metrics : Vec<Box<MetricType>>,
        output_metric : Box<dyn Metric<OutputType>>,    
    ) -> Renegade 
    where 
     InstanceType : Row<InputType, OutputType>,
     TrainingDataType : IntoIterator<Item = InstanceType>,
     MetricType : Metric<InputType> + Labelled, {
         
        Renegade {}
    }

    pub fn learn_metrics<InputType, OutputType, InstanceType, TrainingDataType, MetricType>(
        training_data : TrainingDataType, 
        input_metrics : Vec<Box<MetricType>>,
        output_metric : Box<dyn Metric<OutputType>>,    
    ) -> MetricType 
    where 
     InstanceType : Row<InputType, OutputType>,
     TrainingDataType : IntoIterator<Item = InstanceType>,
     MetricType : Metric<InputType> + Labelled, {
         
        todo!()
    }
}