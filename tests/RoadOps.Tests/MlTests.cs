using RoadOps.ML;

namespace RoadOps.Tests;

public sealed class MlTests
{
    [Fact]
    public void Metrics_are_correct_for_a_small_known_example()
    {
        var result = Metrics.Calculate([RoadCondition.Normal, RoadCondition.Warning, RoadCondition.Critical], [RoadCondition.Normal, RoadCondition.Critical, RoadCondition.Critical]);
        Assert.Equal(2f / 3f, result.Accuracy, 4);
        Assert.Equal(1, result.ConfusionMatrix[(int)RoadCondition.Warning][(int)RoadCondition.Critical]);
        Assert.Equal(1f, result.PerClass[RoadCondition.Critical].Recall, 4);
    }

    [Fact]
    public void Dataset_generation_and_split_are_deterministic()
    {
        var path = Path.Combine(Path.GetTempPath(), $"roadops-{Guid.NewGuid()}", "telemetry.csv");
        var rows = TelemetryDataset.LoadOrCreate(path, 60, 7);
        var first = TelemetryDataset.Split(rows, .2f, 9); var second = TelemetryDataset.Split(rows, .2f, 9);
        Assert.Equal(60, rows.Count);
        Assert.Equal(first.Test.Select(x => x.Telemetry), second.Test.Select(x => x.Telemetry));
        Assert.True(File.Exists(path));
    }

    [Fact]
    public void Normalization_is_fit_only_from_training_rows()
    {
        var train = new[] { new LabeledTelemetry(new(0, 0, 0, 0, 0), RoadCondition.Normal), new LabeledTelemetry(new(10, 10, 10, 10, 10), RoadCondition.Warning) };
        var profile = NormalizationProfile.Fit(train);
        Assert.Equal(5, profile.Means[0]);
        Assert.Equal(3f, profile.Transform(new(20, 20, 20, 20, 20))[0], 3);
    }

    [Fact]
    public void Trained_weights_can_be_loaded_for_prediction()
    {
        var root = Path.Combine(Path.GetTempPath(), $"roadops-model-{Guid.NewGuid()}");
        var service = new MlService(new(Path.Combine(root, "telemetry.csv"), Path.Combine(root, "artifacts"), Epochs: 3));
        service.Train();
        var prediction = service.Predict(new(55, 2, 1, 3, 1));
        Assert.InRange(prediction.Probabilities.Values.Sum(), .99f, 1.01f);
    }
}
