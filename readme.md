## **Project Overview**

The transport industry is a critical component of the global economy. The efficient movement of goods is necessary to ensure that businesses can operate effectively and customers can receive their products on time. However, determining the appropriate mode of transport for each shipment can be challenging. It requires considering various factors such as the type of product being transported, the distance, and the destination.
Choosing the appropriate mode of transport for each shipment can significantly affect the delivery time, cost, and safety of the goods being transported.

For example, air transport is generally faster but more expensive than other modes, while sea transport is slower but more cost-effective for large shipments. The wrong choice of transport mode can result in delays, damage to the goods, or increased costs for the business. By accurately predicting the appropriate mode of transport for each shipment, businesses can optimize their logistics operations, reduce costs, and improve customer satisfaction.


## **Execution Instructions**
<br>
 
### Option 1: Running on your computer locally
 
To run the notebook on your local system set up a [python](https://www.python.org/) environment. Set up the [jupyter notebook](https://jupyter.org/install) with python or by using [anaconda distribution](https://anaconda.org/anaconda/jupyter). Download the notebook and open a jupyter notebook to run the code on local system.
 
The notebook can also be executed by using [Visual Studio Code](https://code.visualstudio.com/), and [PyCharm](https://www.jetbrains.com/pycharm/).

**Python Version: 3.8.10**

* Create a python environment using the command 'python3 -m venv myenv'.

* Activate the environment by running the command 'myenv\Scripts\activate.bat'.

* Install the requirements using the command 'pip install -r requirements.txt'

* Run engine.py with the command 'python3 engine.py'.
 

 
### Option 2: Executing with Colab
Colab, or "Collaboratory", allows you to write and execute Python in your browser, with access to GPUs free of charge and easy sharing.
 
You can run the code using [Google Colab](https://colab.research.google.com/) by uploading the ipython notebook. 
 



## **Data Reading with GCP BigQuery**

Cloud computing has revolutionized the way data is stored, processed, and analyzed. Google Cloud Platform (GCP) provides a suite of tools and services that allow us to manage and analyze large datasets efficiently. One of the most powerful tools provided by GCP is BigQuery, a serverless, highly-scalable, and cost-effective data warehouse that can process and analyze petabyte-scale datasets in real-time. In this project, we will be using BigQuery to read, process, and prepare the data for machine learning modeling. Using BigQuery eliminates the need for setting up a traditional database management system or writing complex code for data processing. By leveraging BigQuery, we can quickly preprocess large datasets and transform them into a format suitable for machine learning models.


To run the project, it is necessary to create a BigQuery table from the CSV file provided in the code folder by uploading it to Cloud Storage. Once the table is created, we can read it using the BigQuery API in Python. To do this, we need to authenticate our email ID and change the project ID in the read_gbq function to the name of the project we saved the table in.


### **Creating a Bucket in Google Cloud Storage**

* To create a bucket in Google Cloud Storage and upload a CSV file into it, you can follow these steps:

* Open the Google Cloud Console and select the project that you want to work with.

* In the left pane, select "Storage" and then "Storage Browser".

* Click on the "Create Bucket" button.

* In the "Create Bucket" dialog box, specify the name of the bucket, choose a location for the bucket, and select the default storage class for the bucket.

* Click on the "Create" button to create the bucket.

* Once the bucket is created, select it in the Storage Browser.

* Click on the "Upload Files" button.

* In the "Upload Files" dialog box, select the CSV file that you want to upload and click on the "Open" button.

* Wait for the upload to complete. Once it is finished, you should see the CSV file listed in the bucket in the Storage Browser.


### **Creating a Dataset in BigQuery with a CSV stored in Cloud Storage**

* To create a table in BigQuery from a CSV file stored in Cloud Storage, you can follow these steps:

* Open the BigQuery web console in the Google Cloud Console.

* Select the project that you want to work with.

* In the left pane, select the dataset where you want to create the table.

* Click on the "Create Table" button.

* In the "Create Table" dialog box, select "Google Cloud Storage" as the "Source".

* Specify the location of your CSV file in Cloud Storage, either by typing the path or using the file picker.

* In the "Schema" section, define the structure of your table by specifying the column names, data types, and any additional properties.

* Click on the "Create Table" button to create the table.

* BigQuery will automatically load the data from the CSV file into the newly created table.

### **Authentication using Local System**

Once you have created the BigQuery table and added a service account (permissions access to an email id),
you can download the GOOGLE_APPLICATION_CREDENTIALS JSON file from the Google Cloud Console.

Here's how you can download it:

* Go to the Google Cloud Console.
* Select your project and go to the "IAM & admin" section from the left sidebar.
* Click on "Service accounts".
* Create a new service account or select an existing one.
* Click on "Create key" and select "JSON" as the key type.
* Save the JSON file to your local machine.


A dummy Access key has been given in the code folder.