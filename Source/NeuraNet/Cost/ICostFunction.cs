using MathNet.Numerics.LinearAlgebra;

namespace NeuraNet.Cost
{
    public interface ICostFunction
    {
        double Calculate(Vector<double> output, Vector<double> target);
    }
}
