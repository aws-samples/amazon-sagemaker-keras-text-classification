# AIM427-Take an ML from idea to production using Amazon SageMaker

## Workshop Lab Guide

Amazon SageMaker has built-in algorithms that let you quickly get started on extracting value out of your data. However, for customers using frameworks or libraries not natively supported by Amazon SageMaker or customers that want to use custom training/inference code, it also offers the capability to package, train and serve the models using custom Docker images.

In this workshop, you will work on this advanced use-case of building, training and deploying ML models using custom built TensorFlow Docker containers.

The model we will develop will classify news articles into the appropriate news category. To train our model, we will be using the [UCI News Dataset](https://archive.ics.uci.edu/ml/datasets/News+Aggregator) which contains a list of about 420K articles and their appropriate categories (labels). There are four categories: Business (b), Science & Technology (t), Entertainment (e) and Health & Medicine (m).

### LAB1: Dataset Exploration

Before we dive into the mechanics of our deep learning model, let’s explore the dataset and see what information we can use to predict the category. For this, we will use a notebook within Amazon SageMaker that we will can also utilize later on as our development machine.

Follow these steps to launch a notebook, download and explore the dataset:

1\. Open the Amazon SageMaker Console, select 'Notebook instances' on the left and then click on ‘Create notebook instance’ and give the notebook a name. For the instance type, I’m going to pick ‘ml.t3.medium’ since our example dataset is small and we don’t intend on using GPUs for training/inference. We're not planning to use Elastic Inference either, so you can leave the default of ‘none’.

For the IAM role, select ‘Create a new role’ and select the options shown below for the role configuration. We don't need access to specific S3 buckets, so you can select ‘None’.

![Amazon SageMaker IAM Role](/images/sm-keras-1.png)

Click ‘Create role’ to create a new role. In the ‘Git repositories’ section select the option to clone a public Git repository and use this URL: https://github.com/aws-samples/amazon-sagemaker-keras-text-classification.git

![Amazon SageMaker Git Repo](/images/sm-keras-git.png)

Hit ‘Create notebook instance’ to submit the request for a new notebook instance.

**Note:** It usually takes a few minutes for the notebook instance to become available. Once available, the status of the notebook instance will change from ‘Pending’ to ‘InService’. You can move on to the next step while notebook instance is still in 'Pending' state.

2\. While the notebook instance comes up, let’s go ahead and add a managed IAM policy to give the notebook instance and Amazon SageMaker access to read and write images to the Elastic Container Repository (ECR) service so that we can push the Docker images from our notebook instance and Amazon SageMaker can retrieve them for training and inference.

From the Amazon SageMaker console, click on the name of the notebook instance you just created:

![SageMaker console instance list](/images/sagemaker-notebook-list.png)

From the notebook instance details page, click on the new role that you just created.

![SageMaker console instance details](/images/sagemaker-notebook-permissions.png)

This will open up a new tab showing the IAM role details. Here click on ‘Attach policies’ and then search for ‘AmazonEC2ContainerRegistryFullAccess’ policy, select it and then click on ‘Attach policy’.

![SageMaker IAM Role Policy](/images/sm-keras-4.png)

*Please make sure to check the checkbox next to the policy before hitting `Attach policy`*

3\.	From the Amazon SageMaker console, click ‘Open Jupyter’ to navigate into the Jupyter notebook. Under ‘New’, select ‘Terminal’. This will open up a terminal session to your notebook instance.

![SageMaker Notebook Terminal](/images/sm-keras-new-terminal.png)

4\.	Switch into the ‘data’ directory

```
cd SageMaker/amazon-sagemaker-keras-text-classification/data
```

5\. Download and unzip the dataset

```
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip && unzip NewsAggregatorDataset.zip
```

6\. Now lets also download and unzip the pre-trained glove embedding files (more on this in a bit):

```
wget http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip
```

7\. Remove the unnecessary files

```
rm 2pageSessions.csv glove.6B.200d.txt glove.6B.50d.txt glove.6B.300d.txt glove.6B.zip readme.txt NewsAggregatorDataset.zip && rm -rf __MACOSX/
```

At this point, you should only see two files: ‘glove.6B.100d.txt’ (word embeddings) and ‘newsCorpora.csv’ (dataset) in the this data directory.

8\. Go back to the Jupyter notebook web UI. You should be in the folder called ‘sagemaker_keras_text_classification’. Please launch the notebook within it with the same name. Make sure the kernel you are running is ‘conda_tensorflow_p36’.

![SageMaker notebook kernel](/images/sagemaker-notebook-kernel.png)

If it’s not, you can switch it from ‘Kernel -> Change kernel’ menu:

![SageMaker notebook change kernel](/images/sagemaker-notebook-kernel-change.png)

9\. Once you individually run the cells within this notebook (shift+enter) through ‘Step 1: Data Exploration’, you should see some sample data (Note: do not run all cells within the notebook – the example is designed to be followed one cell at a time):

![SageMaker notebook data exploration](/images/sm-keras-7.png)

Here we first import the necessary libraries and tools such as TensorFlow, pandas and numpy. An open-source high performance data analysis library, pandas is an essential tool used in almost every Python-based data science experiment. NumPy is another Python library that provides data structures to hold multi-dimensional array data and provides many utility functions to transform that data. TensorFlow is a widely used deep learning framework that also includes the higher-level deep learning Python library called Keras. We will be using Keras to build and iterate our text classification model.

Next we define the list of columns contained in this dataset (the format is usually described as part of the dataset as it is here). Finally, we use the ‘read_csv()’ method of the pandas library to read the dataset into memory and look at the first few lines using the ‘head()’ method.

**Remember, our goal is to accurately predict the category of any news article. So, ‘Category’ is our label or target column. For this example, we will only use the information contained in the ‘Title’ to predict the category.**

### LAB 2: Building the SageMaker TensorFlow Container

Since we are going to be using a custom built container for this workshop, we will need to create it. We will use this container for local testing. Once satisfied with local testing, we will push it up to Amazon Container Registery (ECR) where it can pulled from by Amazon SageMaker for training and deployment.

Instead of building a TensorFlow container from scratch, we are going to use the AWS Deep Learning (DL) Containers. AWS DL Containers are Docker images pre-installed with deep learning frameworks to make it easy to deploy custom machine learning (ML) environments quickly.

AWS DL Containers support TensorFlow, PyTorch, and Apache MXNet. We are going to use TensorFlow today. You can deploy AWS DL Containers on Amazon Sagemaker, Amazon Elastic Kubernetes Service (Amazon EKS), self-managed Kubernetes on Amazon EC2, Amazon Elastic Container Service (Amazon ECS). We are going to deploy using SageMaker for training and inference. The containers are available through Amazon Elastic Container Registry (Amazon ECR) and AWS Marketplace at no cost, you pay only for the resources that you use. In this workshop, we a re going to download the AWS DL Containers images via ECR.

For your reference, all available AWS DL Containers images are described in the documentation:

https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers-images.html

1\. Change directory to be in the path where we are going to create the custom container:

```
cd ~/SageMaker/amazon-sagemaker-keras-text-classification/container/
```

2\. Create a new Dockerfile using `vim Dockerfile`, hit `i` to insert and then paste the content below. In the line starting with `FROM`, replace `REGION` with the AWS region you are using today, for example `us-west-2`. To replace a word with `vim`, hit Escape and then `cw`, finally type the new word.

```
# Build an image that can do training and inference in SageMaker

FROM 763104351884.dkr.ecr.REGION.amazonaws.com/tensorflow-training:1.14.0-cpu-py36-ubuntu16.04

RUN apt-get update && \
    apt-get install -y nginx

RUN pip install gevent gunicorn flask

ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY sagemaker_keras_text_classification /opt/program
WORKDIR /opt/program
```

Hit Escape and then `:wq` to save and exit vim.

We start from the `base` image, add the code directory to our path, copy the code into that directory and finally set the WORKDIR to the same path so any subsequent RUN/ENTRYPOINT commands run by Amazon SageMaker will use this directory.

3\. Build the custom image

Run the docker login command emitted from the command below to log in

```
aws ecr get-login --no-include-email --region <replace with AWS REGION such as us-east-1 or us-west-2> --registry-ids 763104351884
```

and then run following command

```
docker build -t sagemaker-keras-text-class:latest .
```

### LAB 3: Local Testing of Training & Inference Code

Once we are finished developing the training portion (in ‘container/train’), we can start testing locally so we can debug our code quickly. Local test scripts are found in the ‘container/local_test’ subfolder. Here we can run ‘train_local.sh’ which will, in turn, run a Docker container within which our training code will execute.

#### Testing Training Code

1\.	The local testing framework expects the training data to be in the ‘/container/local_test/test_dir/input/data/training’ folder so let’s copy over the contents of our ‘data’ folder there.

In the notebook instance terminal window, switch over to the ‘sagemaker-keras-text-classification/data’ directory

```
cd ~/SageMaker/amazon-sagemaker-keras-text-classification/data
```

 and then run:

```
cp -a . ../container/local_test/test_dir/input/data/training/
```

2\. Switch into the ‘local_test’ directory

```
cd ../container/local_test
```

3\. Run the following command to run the training locally.

```
./train_local.sh sagemaker-keras-text-class:latest
```

*Note:* it might take anywhere from 2-3 minutes to complete for the local training to complete.

![local training results](/images/sagemaker-terminal-local-training.png)

With an 80/20 split between the training and validation and a simple Feed Forward Neural Network, we get around 78-80% validation accuracy (val_acc) after two epochs – not a bad start!

Try reaching 84-85% validation accouracy before going to the next step. You can test different network architectures (using different hyperparameters) by editing `~/SageMaker/amazon-sagemaker-keras-text-classification/container/train` and create more docker containers (each with a unique name) that you can train local (you don't need to edit the Dockerfile). For example:

```
cd ~/SageMaker/amazon-sagemaker-keras-text-classification/container/
vim ./sagemaker_keras_text_classification/train
docker build -t sagemaker-keras-text-class-2units-2layers:latest .
cd ../container/local_test
./train_local.sh sagemaker-keras-text-class-2units-2layers:latest
```

When running training locally multiple times, you should confirm when asked to remove some write-protected regular files. Those files are the model output of the previous training (the ‘news_breaker.h5’ and the ‘tokenizer.pickle’ files).

Here's the part of the `train` file where you can change the network architecture to have more units or add new layers:

```
# ------Architecture: MLP------------------------------
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2, activation='relu')) # Try 2-32 units (dimensionality)
# model.add(tf.keras.layers.Dense(2, activation='relu')) # Try adding more layers uncommenting this line and changing the units
#------------------------------------------------------
```

*Note:* for each test it might take anywhere from 2-3 minutes to complete for the local training to complete, we recommend you do 2-3 tests and them move forward with the best result you got.

If the network architecture is too "small", then it may be incapable of "learn" from the traning data and accourancy cannot increase beyond a certain point. That is a case of underfitting.
If the network architecture is too "complex", it can learn to fit to the training data so very well, but then the model is not capable of working on new data points, so the validation accourancy is much lower than the training accourancy.

We now have a saved model called ‘news_breaker.h5’ and the ‘tokenizer.pickle’ file within ‘sagemaker-keras-text-classification/container/local_test /test_dir/model’ – the local directory that we mapped to the ‘/opt/ml’ directory within the container.

#### Testing Inference Code

It is also advisable to locally test and debug the interference Flask app so we don’t waste time debugging it when we deploy it to Amazon SageMaker.

4\.	Start local testing by running ‘serve_local.sh’ in the ‘local_test’ directory (we should be in it already after completing previous step):

```
./serve_local.sh sagemaker-keras-text-class:latest
```

This is a simple script that uses the ‘Docker run’ command to start the container and the Flask app that we defined previously in the `serve` file.

5\. Now **open another terminal**, move to the `local_test` directory and run ‘predict.sh’. This script issues a request to the flask app using the test news headline in `input.json`:

```
cd SageMaker/amazon-sagemaker-keras-text-classification/container/local_test && cat input.json && ./predict.sh input.json application/json
```

Great! Our model inference implementation responds and is correctly able to categorize this headline as a Health & Medicine story.

### Lab 4: Training & Deployment on Amazon SageMaker

Now that we are done testing locally, we are ready to package up our code and submit to Amazon SageMaker for training or deployment (hosting) or both.

1\.	We should probably modify our training code to take advantage of the more powerful hardware. Let’s update the number of epochs in the ‘train’ script to `10` to see how that impacts the validation accuracy of our model while training on Amazon SageMaker. This file is located in 'sagemaker_keras_text_classification' directory. Navigate there by
```
cd ../sagemaker_keras_text_classification/
```
and edit the file named 'train'

```python
history = model.fit(x_train, y_train,
                            epochs=10,
                            batch_size=32,
                            validation_data=(x_test, y_test))

```

2\. Open the ‘sagemaker_keras_text_classification.ipynb’ notebook and follow the steps listed in **Lab 4** to upload the data to S3, submit the training job and, finally, deploy the model for inference. The notebook contains explanations for each step and also shows how to test your inference endpoint.

### Lab 5: Distributed Training in Script Mode with Horovod Distributed Training Framework.
This lab will demonstrate both SageMaker Horovod framework as well as SageMaker's Script mode.

#### A. SageMaker's Horovod Distribugted Training Framework

Amazon SageMaker has built-in training algorithms that provide the ability to do distributed training among multiple compute nodes via 'train_instance_count' parameter. Up until now, if one were to bring his/her own algorithm, they would have to take care of providing their own distributed compute framework and encapsulating it in a container. Horovod has previously been enabled on Amazon Deep Learning AMIs
(https://aws.amazon.com/blogs/machine-learning/aws-deep-learning-amis-now-include-horovod-for-faster-multi-gpu-tensorflow-training-on-amazon-ec2-p3-instances/).
With the introduction of the new feature described in this lab, Horovod distributed framework is available as part of SageMaker Deep Learning Containers. Customers can run fully-managed Horovod jobs for high scale distributed training.

What is Horovod? It is a framework allowing a user to distribute a deep learning workload among multiple compute nodes and take advantage of inherent parallelism of deep learning training process. It is available for both CPU and GPU AWS compute instances. Horovod follows the Message Passing Interface (MPI) model. This is a popular standard for passing messages and managing communication between nodes in a high-performance distributed computing environment. Horovod’s MPI implementation provides a more simplified programming model compared to the parameter server based distributed training model. This model enables developers to easily scale their existing single CPU/GPU training programs with minimal code changes.

For this lab, we will be instantiating CPU compute nodes for simplicity and scalability.

#### B. SageMaker's Script Mode.
Previously (as in Lab 2-4 of this workshop), in BringYourOwnContainer situation, a user had to make his/her training Python script a part of the container. Therefore, during the debug process, every Python script change required rebuilding the container. SageMaker's "script mode" allows one to build the container once and then debug and change a python script  without rebuilding the container with every change. Instead, a user specifies script's "entry point" via 'train(script="myscript.py",....) parameter, for example:
```
train(script="myscript_train_hvd.py",
        instance_type="ml.c4.xlarge",
        sagemaker_local_session=sage_session,
        docker_image=docker_container_image,
        training_data_path={'train': train_input, 'test': test_input},
        source_dir=source_dir,
        train_instance_count=2,
        base_job_name="tf-hvdwdwdw-cpu",
        hyperparameters={"sagemaker_mpi_enabled": "True"})
```
#### Lab Instructions:
1.\ Open the ‘sentiment-analysis.ipynb’ notebook located in "horovod distributed training" directory and follow its flow. 


## Citations

Dataset: Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

Glove Embeddings: Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. [GloVe: Global Vectors for Word Representation.](https://nlp.stanford.edu/pubs/glove.pdf) [[pdf](https://nlp.stanford.edu/pubs/glove.pdf)] [[bib](https://nlp.stanford.edu/pubs/glove.bib)]



## License

This library is licensed under the Apache 2.0 License.
