def calculate_momentum(graph_centrality, talent_net_flow, news_sentiment_score=0.5):
    """
    Calculates a startup momentum score.
    """
    # Normalize inputs and compute a weighted sum
    w1, w2, w3 = 0.4, 0.4, 0.2
    
    momentum = (w1 * graph_centrality) + (w2 * max(0, talent_net_flow)) + (w3 * news_sentiment_score)
    return momentum
