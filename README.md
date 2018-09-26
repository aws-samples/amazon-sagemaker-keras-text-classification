# AIM410 Workshop Lab Guide: Build, Train and Deploy ML Models using Amazon SageMaker

Amazon SageMaker has built-in algorithms that let you quickly get started on extracting value out of your data. However, for customers using frameworks that are not natively supported by Amazon SageMaker or those that want to use custom training/inference code, it also offers the capability to package, train and serve the custom models using Docker images.

In this workshop, you will work on this advanced use-case of using your own models (utilizing Docker images) and learn how you can take any dataset, train a custom model using Keras (with a TensorFlow backend) and then deploy the model using Amazon SageMaker.

## LAB1: Dataset Exploration

The model we will develop will classify news articles into the appropriate news category. To train our model, we will be using the UCI News Dataset which contains a list of about 420K articles and their appropriate categories (label). There are four categories: Business (b), Science & Technology (t), Entertainment (e) and Health & Medicine (m).

Before we dive into the mechanics of our deep learning model, let’s explore the dataset and see what information we can use to predict the category. For this, we will use a notebook within Amazon SageMaker that we will can also utilize later on as our development machine.

Follow these steps to launch a notebook, download and explore the dataset:

1\.	Open the Amazon SageMaker Console, click on ‘Create notebook instance’ and give the notebook a name. For the instance type, I’m going to pick ‘ml.t2.medium’ since our example dataset is small and I don’t intend on using the GPUs for training/inference.

For the IAM role, select ‘Create a new role’ and select the options shown below for the role configuration.

![Amazon SageMaker IAM Role](/images/sm-keras-1.png)

Click ‘Create role’ to create a new role and then hit ‘Create notebook instance’ to submit the request for a new notebook instance.

**Note:** It usually takes a few minutes for the notebook instance to become available. Once available, the status of the notebook instance will change from ‘Pending’ to ‘InService’. You can move on to the next step while notebook instance is still in 'Pending' state.

2\. While the notebook instance comes up, let’s go ahead and add a managed IAM policy to give the notebook instance and Amazon SageMaker access to read and write images to the Elastic Container Repository (ECR) service so that we can push the Docker images from our notebook instance and Amazon SageMaker can retrieve them for training and inference.

From the Amazon SageMaker console, click on the name of the notebook instance you just created:

![SageMaker console instance list](/images/sm-keras-2.png)



## License

This library is licensed under the Apache 2.0 License.
