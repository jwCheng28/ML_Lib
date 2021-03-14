ADD_LIBRARY(
    InstanceBased STATIC 
    InstanceBased/knn.cpp
)

ADD_LIBRARY(
    LinearModel STATIC
    LinearModel/linreg.cpp
    LinearModel/logreg.cpp
)
