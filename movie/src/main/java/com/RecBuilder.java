package com;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

public class RecBuilder implements RecommenderBuilder {

    @Override
    public Recommender buildRecommender(DataModel model) throws TasteException {
        UserSimilarity userSimilarity = new PearsonCorrelationSimilarity(model);
        UserNeighborhood userNeighborhood = new ThresholdUserNeighborhood(0.1, userSimilarity, model);
        return new GenericUserBasedRecommender(model, userNeighborhood, userSimilarity);
    }
}
