﻿using System.Collections.Generic;
using System.Linq;

using FluentAssertions;

using NeuraNet.Activations;
using NeuraNet.Cost;
using NeuraNet.NetworkLayout;
using NeuraNet.Specs.Builders;

using Xunit;

namespace NeuraNet.Specs
{
    public class NeuralNetworkSpecs
    {
        private class TwoLayerNetworkProvider : NetworkLayoutProvider
        {
            private readonly ILayerInitializer hiddenLayerInitializer = new LayerInitializerBuilder()
                .UsingWeights(new[,]
                {
                    { 0.001, 0.002, 0.003, 0.004 },
                    { 0.005, 0.006, 0.007, 0.008 },
                    { 0.009, 0.010, 0.011, 0.012 }
                })
                .UsingBiases(new[]
                {
                    0.013,
                    0.014,
                    0.015,
                    0.016
                })
                .Build();

            private readonly ILayerInitializer outputLayerInitializer = new LayerInitializerBuilder()
                .UsingWeights(new[,]
                {
                    { 0.017, 0.018 },
                    { 0.019, 0.020 },
                    { 0.021, 0.022 },
                    { 0.023, 0.024 }
                })
                .UsingBiases(new[]
                {
                    0.025,
                    0.026
                })
                .Build();

            public TwoLayerNetworkProvider()
            {
                layers = new List<Layer>
                {
                    new Layer(3, 4, hiddenLayerInitializer, new SigmoidActivation()),
                    new Layer(4, 2, outputLayerInitializer, new SigmoidActivation())
                };

                ConnectLayers();
            }
        }

        [Fact]
        public void When_querying_a_network_that_has_known_weights_and_biases_it_should_yield_the_expected_output()
        {
            // Arrange
            var network = new NeuralNetworkBuilder()
                .Using(new TwoLayerNetworkProvider())
                .Build();

            // Act
            double[] networkOutput = network.Query(new[] { 1.0, -2.0, 3.0 });

            // Assert
            networkOutput[0].Should().BeApproximately(0.5164, 0.00005);
            networkOutput[1].Should().BeApproximately(0.5172, 0.00005);
        }

        [Fact]
        public void When_training_the_network_for_a_single_epoch_it_should_return_the_cost()
        {
            // Arrange
            var network = new NeuralNetworkBuilder()
                .Using(new TwoLayerNetworkProvider())
                .Using(new QuadraticCost())
                .Build();

            // Act
            double cost = network.Train(new[]
            {
                new TrainingExample(new[] { 1.0, -2.0, 3.0 }, new[] { 0.1234, 0.8766 })
            }, 1);

            // Assert
            cost.Should().BeApproximately(0.1418, 0.00005);
        }

        [Fact]
        public void When_training_the_network_for_a_single_epoch_it_should_calculate_the_gradient_vectors()
        {
            // Arrange
            var network = new NeuralNetworkBuilder()
                .Using(new TwoLayerNetworkProvider())
                .Build();

            // Act
            network.Train(new[]
            {
                new TrainingExample(new[] { 1.0, -2.0, 3.0 }, new[] { 0.1234, 0.8766 })
            }, 1);

            // Assert
            Layer hiddenLayer = network.GetLayers().First();
            Layer outputLayer = network.GetLayers().Last();

            double[] outputWeightGradients = outputLayer.WeightGradients.ToColumnMajorArray();
            outputWeightGradients[0].Should().BeApproximately(0.04983553, 0.000000005);
            outputWeightGradients[1].Should().BeApproximately(0.04990912, 0.000000005);
            outputWeightGradients[2].Should().BeApproximately(0.04998271, 0.000000005);
            outputWeightGradients[3].Should().BeApproximately(0.05005629, 0.000000005);
            outputWeightGradients[4].Should().BeApproximately(-0.04556976, 0.000000005);
            outputWeightGradients[5].Should().BeApproximately(-0.04563706, 0.000000005);
            outputWeightGradients[6].Should().BeApproximately(-0.04570435, 0.000000005);
            outputWeightGradients[7].Should().BeApproximately(-0.04577163, 0.000000005);

            var outputBiasGradients = outputLayer.BiasGradients.ToArray();
            outputBiasGradients[0].Should().BeApproximately(0.098150, 0.0000005);
            outputBiasGradients[1].Should().BeApproximately(-0.089749, 0.0000005);

            var outputInputGradients = outputLayer.PreviousLayerActivationGradients.ToArray();
            outputInputGradients[0].Should().BeApproximately(0.00005307, 0.000000005);
            outputInputGradients[1].Should().BeApproximately(0.00006988, 0.000000005);
            outputInputGradients[2].Should().BeApproximately(0.00008668, 0.000000005);
            outputInputGradients[3].Should().BeApproximately(0.00010348, 0.000000005);

            double[] hiddenWeightGradients = hiddenLayer.WeightGradients.ToColumnMajorArray();
            hiddenWeightGradients[0].Should().BeApproximately(0.00001326, 0.0000005);
            hiddenWeightGradients[1].Should().BeApproximately(-0.00002653, 0.0000005);
            hiddenWeightGradients[2].Should().BeApproximately(0.00003980, 0.0000005);
            hiddenWeightGradients[3].Should().BeApproximately(0.00001746, 0.0000005);
            hiddenWeightGradients[4].Should().BeApproximately(-0.00003493, 0.0000005);
            hiddenWeightGradients[5].Should().BeApproximately(0.00005239, 0.0000005);
            hiddenWeightGradients[6].Should().BeApproximately(0.00002166, 0.0000005);
            hiddenWeightGradients[7].Should().BeApproximately(-0.00004332, 0.0000005);
            hiddenWeightGradients[8].Should().BeApproximately(0.00006499, 0.0000005);
            hiddenWeightGradients[9].Should().BeApproximately(0.00002586, 0.0000005);
            hiddenWeightGradients[10].Should().BeApproximately(-0.00005172, 0.0000005);
            hiddenWeightGradients[11].Should().BeApproximately(0.00007758, 0.0000005);

            double[] hiddenBiasGradients = hiddenLayer.BiasGradients.ToArray();
            hiddenBiasGradients[0].Should().BeApproximately(0.000013265, 0.0000000005);
            hiddenBiasGradients[1].Should().BeApproximately(0.000017464, 0.0000000005);
            hiddenBiasGradients[2].Should().BeApproximately(0.000021662, 0.0000000005);
            hiddenBiasGradients[3].Should().BeApproximately(0.000025860, 0.0000000005);

            hiddenLayer.PreviousLayerActivationGradients.Should().BeNull();
        }
    }
}