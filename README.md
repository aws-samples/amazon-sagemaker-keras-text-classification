# AIM410: Build, Train and Deploy ML Models using Amazon SageMaker

## Workshop Lab Guide

Amazon SageMaker has built-in algorithms that let you quickly get started on extracting value out of your data. However, for customers using frameworks or libraries not natively supported by Amazon SageMaker or customers that want to use custom training/inference code, it also offers the capability to package, train and serve the models using custom Docker images.

In this workshop, you will work on this advanced use-case of building, training and deploying ML models using custom built TensorFlow Docker containers.

The model we will develop will classify news articles into the appropriate news category. To train our model, we will be using the [UCI News Dataset](https://archive.ics.uci.edu/ml/datasets/News+Aggregator) which contains a list of about 420K articles and their appropriate categories (labels). There are four categories: Business (b), Science & Technology (t), Entertainment (e) and Health & Medicine (m).

### LAB1: Dataset Exploration

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

From the notebook instance details page, click on the new role that you just created.

![SageMaker console instance details](/images/sm-keras-3.png)

This will open up a new tab showing the IAM role details. Here click on ‘Attach policies’ and then search for ‘AmazonEC2ContainerRegistryFullAccess’ policy, select it and then click on ‘Attach policy’.

![SageMaker IAM Role Policy](/images/sm-keras-4.png)

*Please make sure to check the checkbox next to the policy before hitting `Attach policy`*

3\.	From the Amazon SageMaker console, click ‘Open’ to navigate into the Jupyter notebook. Under ‘New’, select ‘Terminal’. This will open up a terminal session to your notebook instance.

![SageMaker Notebook Terminal](/images/sm-keras-0.png)

4\. The companion code to this blogpost is on GitHub so let’s go ahead and clone that:

```
git clone https://github.com/aws-samples/amazon-sagemaker-keras-text-classification.git ./SageMaker/sagemaker-keras-text-classification
```
5\.	Switch into the ‘data’ directory

```
cd SageMaker/sagemaker-keras-text-classification/data
```

6\. Download and unzip the dataset

```
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip && unzip NewsAggregatorDataset.zip
```

7\. Now lets also download and unzip the pre-trained glove embedding files (more on this in a bit):

```
wget http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip
```

8\. Remove the unnecessary files

```
rm 2pageSessions.csv glove.6B.200d.txt glove.6B.50d.txt glove.6B.300d.txt glove.6B.zip readme.txt NewsAggregatorDataset.zip && rm -rf __MACOSX/
```

At this point, you should only see two files: ‘glove.6B.100d.txt’ (word embeddings) and ‘newsCorpora.csv’ (dataset) in the this data directory.

9\.	Close the terminal window and go back to the Jupyter notebook web UI. Click on the folder called ‘sagemaker_keras_text_classification’ and launch the notebook within it with the same name. Make sure the kernel you are running is ‘conda_tensforflow_p27’.

![SageMaker notebook kernel](/images/sm-keras-5.png)

If it’s not, you can switch it from ‘Kernel -> Change kernel’ menu:

![SageMaker notebook change kernel](/images/sm-keras-6.png)


10\.	Once you individually run the cells within this notebook (shift+enter) through ‘Step 1: Data Exploration’, you should see some sample data (Note: do not run all cells within the notebook – the example is designed to be followed one cell at a time):

![SageMaker notebook data exploration](/images/sm-keras-7.png)

Here we first import the necessary libraries and tools such as TensorFlow, pandas and numpy. An open-source high performance data analysis library, pandas is an essential tool used in almost every Python-based data science experiment. NumPy is another Python library that provides data structures to hold multi-dimensional array data and provides many utility functions to transform that data. TensorFlow is a widely used deep learning framework that also includes the higher-level deep learning Python library called Keras. We will be using Keras to build and iterate our text classification model.

Next we define the list of columns contained in this dataset (the format is usually described as part of the dataset as it is here). Finally, we use the ‘read_csv()’ method of the pandas library to read the dataset into memory and look at the first few lines using the ‘head()’ method.

**Remember, our goal is to accurately predict the category of any news article. So, ‘Category’ is our label or target column. For this example, we will only use the information contained in the ‘Title’ to predict the category.**

### LAB 2: Building the SageMaker TensorFlow Container

Since we are going to be using a custom built container for this workshop, we will need to create it. The Amazon SageMaker notebook instance already comes loaded with Docker. The SageMaker team has also created the [`sagemaker-tensorflow-container`](https://github.com/aws/sagemaker-tensorflow-container) project that makes it super easy for us to build custom TensorFlow containers that are optimized to run on Amazon SageMaker. Similar containers are also available for other widely used ML/DL frameworks as well.

We will first create a `base` TensorFlow container and then add our custom code to create a `final` container. We will use this `final` container for local testing. Once satisfied with local testing, we will push it up to Amazon Container Registery (ECR) where it can pulled from by Amazon SageMaker for training and deployment.

1\. Let's start by creating the base TensorFlow container. Go to the notebook instance terminal window, switch to the home directory and clone the `sagemaker-tensorflow-container` repo:

```
cd ~
```

```
git clone https://github.com/aws/sagemaker-tensorflow-container.git
```

2\. We will be using TensorFlow 1.8.0 so lets switch to the appropriate directory

```
cd sagemaker-tensorflow-container/docker/1.8.0/base
```

3\. If you list the directory contents here, you will notice that there are two Dockerfiles - one made for CPU based nodes and another for GPU based. Since, we will be using CPU machines, lets build the CPU docker image

```
docker build -t tensorflow-base:1.8.0-cpu-py2 -f Dockerfile.cpu .
```

Building the docker images should not take more than 5-7 minutes. Once finished, you can list the images by running `docker images`. You should see the new base image named `tensorflow-base:1.8.0-cpu-py2`.

4\. Next we create our `final` images by including our code onto the `base` container. In the terminal window, switch to the container directory

```
cd ~/SageMaker/sagemaker-keras-text-classification/container/
```

5\. Create a new Dockerfile using `vim Dockerfile`, hit `i` to insert and then paste the content below

```
# Build an image that can do training and inference in SageMaker

FROM tensorflow-base:1.8.0-cpu-py2

ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY sagemaker_keras_text_classification /opt/program
WORKDIR /opt/program
```
Hit Escape and then `:wq` to save and exit vim.

We start from the `base` image, add the code directory to our path, copy the code into that directory and finally set the WORKDIR to the same path so any subsequent RUN/ENTRYPOINT commands run by Amazon SageMaker will use this directory.

6\. Build the `final` image

```
docker build -t sagemaker-keras-text-class:latest .
```

### LAB 3: Local Testing of Training & Inference Code

Once we are finished developing the training portion (in ‘container/train’), we can start testing locally so we can debug our code quickly. Local test scripts are found in the ‘container/local_test’ subfolder. Here we can run ‘local_train.sh’ which will, in turn, run a Docker container within which our training code will execute.

#### Testing Training Code

1\.	The local testing framework expects the training data to be in the ‘/container/local_test/test_dir/input/data/training’ folder so let’s copy over the contents of our ‘data’ folder there.

In the notebook instance terminal window, switch over to the ‘sagemaker-keras-text-classification/data’ directory

```
cd ~/SageMaker/sagemaker-keras-text-classification/data
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

With an 80/20 split between the training and validation and a simple Feed Forward Neural Network, we get around 85% validation accuracy after two epochs – not a bad start!

![local training results](/images/sm-keras-8.png)

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
cd SageMaker/sagemaker-keras-text-classification/container/local_test && ./predict.sh input.json application/json
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
With the introduction of the new feature described in this lab, Horovod distributed framework is available as part of SageMaker BringYourOwnContainer flow. Customers will soon also be able to run fully-managed Horovod jobs for high scale distributed training.

What is Horovod? It is a framework allowing a user to distribute a deep learning workload among multiple compute nodes and take advantage of inherent parallelism of deep learning training process. It is available for both CPU and GPU AWS compute instances. Horovod follows the Message Passing Interface (MPI) model. This is a popular standard for passing messages and managing communication between nodes in a high-performance distributed computing environment. Horovod’s MPI implementation provides a more simplified programming model compared to the parameter server based distributed training model. This model enables developers to easily scale their existing single CPU/GPU training programs with minimal code changes.

For this lab, we will be instantiating CPU compute nodes for simplicity and scalability.

#### B. SageMaker's Script Mode.
Previously (as in Lab 2-4 of this workshop), in BringYourOwnContainer situation, a user had to make his/her training Python script a part of the container. Therefore, during the debug process, every Python script change required rebuilding the container. SageMaker's "script mode" allows one to build the container once and then debug and change a python script  without rebuilding the container with every change. Instead, a user specifies script's "entry point" via 'train(script="myscript.py",....) parameter, for example:
```
train(script="train_mnist_hvd.py",
        instance_type="ml.c4.xlarge",
        sagemaker_local_session=sage_session,
        docker_image=docker_container_image,
        training_data_path={'train': train_input, 'test': test_input},
        source_dir=source_dir,
        train_instance_count=8,
        base_job_name="tf-hvdwdwdw-cpu",
        hyperparameters={"sagemaker_mpi_enabled": "True"})
```
#### Lab Instructions:
1\.	From the Amazon SageMaker console, click ‘Open’ to navigate into the Jupyter notebook. Under ‘New’, select ‘Terminal’. This will open up a terminal session to your notebook instance.

![SageMaker Notebook Terminal](/images/sm-keras-0.png)


2\. Switch to the `~/SageMaker` directory and clone the distributed training code
```
cd ~/SageMaker
```
```
git clone https://github.com/aws-samples/SageMaker-Horovod-Distributed-Training
```


3\. Open the ‘Tensorflow Distributed Training - Horovod-BYOC.ipynb’ notebook located in SageMaker-Horovod-Distributed-Training directory and follow its flow. This will unpack the container with keras-mnist dataset and execute a distributed training on 8 CPU nodes.


## Citations

Dataset: Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

Glove Embeddings: Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. [GloVe: Global Vectors for Word Representation.](https://nlp.stanford.edu/pubs/glove.pdf) [[pdf](https://nlp.stanford.edu/pubs/glove.pdf)] [[bib](https://nlp.stanford.edu/pubs/glove.bib)]



## License

This library is licensed under the Apache 2.0 License.
