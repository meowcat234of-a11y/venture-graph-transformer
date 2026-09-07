def calculate_momentum(graph_centrality, talent_net_flow, news_sentiment_score=0.5):
    if graph_centrality < 0 or not 0 <= news_sentiment_score <= 1:
        raise ValueError("centrality must be non-negative and sentiment must be in [0, 1]")
    return 0.4 * graph_centrality + 0.4 * max(0.0, talent_net_flow) + 0.2 * news_sentiment_score
