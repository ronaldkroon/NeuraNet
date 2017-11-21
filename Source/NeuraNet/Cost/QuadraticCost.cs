using MathNet.Numerics.LinearAlgebra;

namespace NeuraNet.Cost
{
    public class QuadraticCost : ICostFunction
    {
        /// <summary>
        /// The training algorithm has done a good job if it can find weights and biases for which the quadratic cost
        /// is close to 0. It's not doing so well when the cost is large, because that would mean that
        /// networkOutput - targetOutput is not close to the output for a large number of inputs.
        /// </summary>
        public double Calculate(Vector<double> output, Vector<double> target)
        {
            double Squared(double value) => (value * value);

            return 0.5 * ((output - target).Map(Squared).Sum());
        }
    }
}