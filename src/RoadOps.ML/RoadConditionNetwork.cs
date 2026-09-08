using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace RoadOps.ML;

public sealed class RoadConditionNetwork : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _layers = Sequential(("input", Linear(5, 16)), ("relu", ReLU()), ("output", Linear(16, 3)));
    public RoadConditionNetwork() : base(nameof(RoadConditionNetwork)) => RegisterComponents();
    public override Tensor forward(Tensor input) => _layers.forward(input);
}
