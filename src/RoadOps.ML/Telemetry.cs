namespace RoadOps.ML;

public enum RoadCondition { Normal = 0, Warning = 1, Critical = 2 }

public sealed record Telemetry(float VehicleSpeed, float AmbientTemperature, float Rainfall, float RoadTemperature, float Vibration)
{
    public float[] Features => [VehicleSpeed, AmbientTemperature, Rainfall, RoadTemperature, Vibration];
}

public sealed record LabeledTelemetry(Telemetry Telemetry, RoadCondition Label);

public sealed record NormalizationProfile(float[] Means, float[] StandardDeviations)
{
    public static NormalizationProfile Fit(IReadOnlyList<LabeledTelemetry> rows)
    {
        var means = Enumerable.Range(0, 5).Select(i => rows.Average(r => r.Telemetry.Features[i])).Select(x => (float)x).ToArray();
        var std = Enumerable.Range(0, 5).Select(i => MathF.Sqrt(rows.Average(r => MathF.Pow(r.Telemetry.Features[i] - means[i], 2)))).ToArray();
        return new(means, std.Select(x => x < 0.00001f ? 1f : x).ToArray());
    }

    public float[] Transform(Telemetry telemetry) => telemetry.Features.Select((x, i) => (x - Means[i]) / StandardDeviations[i]).ToArray();
}
