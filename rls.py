import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score




if __name__ == '__main__':
    
    df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv")
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]

    train_X = np.asanyarray(train[['ENGINESIZE']])
    test_X = np.asanyarray(test[['ENGINESIZE']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])

    regr = linear_model.LinearRegression()
    regr.fit(train_X,train_y)
    preds = regr.predict(test_X)

    print('Mean Absolute Error: %.2f'% np.mean(np.absolute(preds-test_y)))
    print('Mean Squared Error: %.2f'% np.mean((preds-test_y)**2))
    print('R2 squared: %.2f'% r2_score(test_y,preds))

    plt.scatter(df.ENGINESIZE,df.CO2EMISSIONS,color='blue')
    plt.plot(train_X,regr.coef_*train_X + regr.intercept_, '-r')
    plt.xlabel('Engine Size')
    plt.ylabel('CO2 Emissions')
    plt.show()