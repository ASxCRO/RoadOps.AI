using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace RoadOps.ML;

public sealed record MlOptions(string DataPath, string ArtifactDirectory, int Epochs = 100, float LearningRate = .01f);
public sealed record TrainingResult(ClassificationMetrics Metrics, int TrainRows, int TestRows);
public sealed record Prediction(RoadCondition Condition, IReadOnlyDictionary<RoadCondition, float> Probabilities);
internal sealed record SavedModel(NormalizationProfile Profile, string SchemaVersion = "1");

public sealed class MlService(MlOptions options)
{
    private readonly string _weights = Path.Combine(options.ArtifactDirectory, "road-condition.pt");
    private readonly string _metadata = Path.Combine(options.ArtifactDirectory, "road-condition.json");

    public TrainingResult Train()
    {
        torch.random.manual_seed(42);
        var rows = TelemetryDataset.LoadOrCreate(options.DataPath); var (train, test) = TelemetryDataset.Split(rows);
        // Fit feature means/std-devs on training rows only; fitting on test rows would leak evaluation information.
        var profile = NormalizationProfile.Fit(train);
        using var model = new RoadConditionNetwork(); using var optimizer = torch.optim.Adam(model.parameters(), options.LearningRate); using var loss = torch.nn.CrossEntropyLoss();
        // Features are [row, five telemetry features]; labels are class indices [row], as CrossEntropyLoss expects.
        using var features = TensorFrom(train, profile); using var labels = tensor(train.Select(x => (long)x.Label).ToArray(), dtype: ScalarType.Int64);
        // train() enables training behaviour; this small model has no dropout, but keeping it explicit teaches the normal PyTorch lifecycle.
        model.train();
        for (var epoch = 0; epoch < options.Epochs; epoch++)
        {
            // Adam clears previous gradients, backpropagates CrossEntropyLoss over raw class logits, then updates weights.
            optimizer.zero_grad(); using var output = model.forward(features); using var value = loss.forward(output, labels); value.backward(); optimizer.step();
        }
        model.eval();
        // no_grad() stops autograd bookkeeping during inference and evaluation.
        using var guard_ = torch.no_grad(); using var testFeatures = TensorFrom(test, profile); using var logits = model.forward(testFeatures); using var indices = logits.argmax(1);
        var predictions = indices.data<long>().Select(x => (RoadCondition)x).ToArray();
        var metrics = Metrics.Calculate(test.Select(x => x.Label).ToArray(), predictions);
        // Weights alone are insufficient: inference must reuse the exact training normalization profile.
        Directory.CreateDirectory(options.ArtifactDirectory); model.save(_weights); File.WriteAllText(_metadata, JsonSerializer.Serialize(new SavedModel(profile)));
        return new(metrics, train.Count, test.Count);
    }

    public Prediction Predict(Telemetry input)
    {
        if (!File.Exists(_weights) || !File.Exists(_metadata)) throw new InvalidOperationException("No trained ML artifact exists. Call POST /ml/train first.");
        var saved = JsonSerializer.Deserialize<SavedModel>(File.ReadAllText(_metadata)) ?? throw new InvalidOperationException("ML metadata is invalid.");
        using var model = new RoadConditionNetwork(); model.load(_weights); model.eval();
        // A single inference row still has batch shape [1, 5]; softmax only converts logits to displayable probabilities.
        using var guard_ = torch.no_grad(); using var tensor = TensorFrom([new(input, RoadCondition.Normal)], saved.Profile); using var logits = model.forward(tensor); using var probabilities = logits.softmax(1);
        var scores = probabilities.data<float>().ToArray(); var selected = Array.IndexOf(scores, scores.Max());
        return new((RoadCondition)selected, Enum.GetValues<RoadCondition>().ToDictionary(x => x, x => scores[(int)x]));
    }

    private static Tensor TensorFrom(IReadOnlyList<LabeledTelemetry> rows, NormalizationProfile profile) => tensor(rows.SelectMany(x => profile.Transform(x.Telemetry)).ToArray(), dtype: ScalarType.Float32).reshape(rows.Count, 5);
}
