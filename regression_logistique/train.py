import dataset 
import regression_logistique


dataset = dataset.Dataset('../data/train.csv')
logist = regression_logistique.LogisticRegression(*dataset.train)

logist.train(plot_loss=False)

