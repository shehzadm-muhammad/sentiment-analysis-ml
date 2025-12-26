from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

samples = [
    "Amazing service and very friendly staff",
    "Terrible experience, I will not come back",
    "The app is easy to use and works perfectly",
    "Delivery was late and the order was wrong",
]

def label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"

print("\n✅ Sentiment Predictions (VADER):")
for s in samples:
    score = analyzer.polarity_scores(s)["compound"]
    print(f"- {s}  -->  {label(score)} (compound={score:.3f})")

