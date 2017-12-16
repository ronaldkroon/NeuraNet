using MathNet.Numerics.LinearAlgebra;

namespace NeuraNet.Cost
{
    public class QuadraticCost : ICostFunction
    {
        public double Calculate(Vector<double> output, Vector<double> target)
        {
            double Squared(double value) => (value * value);

            return 0.5 * ((output - target).Map(Squared).Sum());
        } 
        
        public Vector<double> Derivative(Vector<double> output, Vector<double> target)
        {
            return output - target;
        }
    }
}