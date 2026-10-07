def item_cf_predict(user_ratings: list, item_similarities: list, target: int) -> float:
    """
    Returns the similarity-weighted rating prediction.
    """
    sum_num = 0
    sum_dem = 0
    for i in range(len(user_ratings)):
        if i==target:
            continue
        if user_ratings[i]>0 and item_similarities[i]>0:
            sum_num += user_ratings[i]*item_similarities[i]
            sum_dem += item_similarities[i]
    if sum_dem==0:
        return 0
    return sum_num/sum_dem