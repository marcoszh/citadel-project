# citadel-project

This repo contains the source codes for the Citadel project appeared in [SoCC 2021](https://dl.acm.org/doi/10.1145/3472883.3486998).

** Note: Citadel is built atop [Scone SGX Containers](https://scontain.com/index.html?lang=en), you will need to have access to their proprietary TensorFlow images to run the codes inside this repo.

Each of the workload requires the input data and packaged docker image to run. I am not authorized to provide you access to these docker images unfortunately.

Application: blindness detection
base image: marcoszh/private_repo:tensorflow
data: https://www.kaggle.com/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered


Application: credit card fraud
base image: marcoszh/private_repo:credit_card
data: https://www.kaggle.com/mlg-ulb/creditcardfraud


Application: spam or ham
base image: marcoszh/private_repo:sms
data: https://www.kaggle.com/uciml/sms-spam-collection-dataset


