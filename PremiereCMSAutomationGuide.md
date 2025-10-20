# PREMIERE Complete Automatic VIDEO Processing from CMS using crontab

Conguration file:
------------
We have notification feature  in this automation pipeline that needs to be configured first, where a notification is send to a user email after a new video being uploaded/processed. 

Therefore, in the config.yaml file add your email and password from which the email is to be sent along with the smtp server details. Also add the email of the receivers (multiple if required).
```
email:
  from_addr: "sample@univ-st-etienne.fr"
  to_addrs:
    - "sample1_recipient@univ-st-etienne.fr"
    - "sample2_recipient@example.com"
    - "sample3_recipient@example.com"
  smtp_server: "smpt server address here"
  smtp_port: <port for your smtp server>
  password: "<Your_Password_of_your_email_here>"
```


Setting up env variables for the digitalocean S3bucket
------------
Go to the URL: https://www.couleur.org/


Username: premiere


Write the required password for this. Then you will find the exporting_aws.txt file that includes the keys for all the variables below which will allow to access the s3 bucket. Setup the environment varialbes

```
export DO_SPACES_KEY="DO_SPACES_KEY"
export DO_SPACES_SECRET="DO_SPACES_SECRET"
export DO_SPACES_REGION="DO_SPACES_REGION"
export DO_SPACES_BUCKET="bucketname"
export DO_SPACES_ENDPOINT="<URL for digital ocean space>"
```

Executing crontab
------------

Type the command in terminal :

```
crontab -e
```
Add the command in the editor opened  along with the time to execute
```
minute hour day month day_of_week cd <path_of_your_directory> && <conda_environment_path> run -n <conda_environment_name>  <script_name>
```

For example: 
```
*/1 * * * * cd /home/user/PMTOOLV2/PREMIERETOOLV2/ && /home/user/miniconda3/bin/conda run -n premiere python gql_client_proxy_v3.py
```
In the above example the script will be executed after every 1 minute.


If needed to terminate the cront tab execution:
```
crontab -r
```
Also be sure to kill the python execution after terminating the crontab session


Results Folder directory
------------
There will be two folders generated once a video has been processed:
* pcloudresults: This contains the processed data results
* pcloudvideos: This contains the videos that has been uploaded

The annotation from this result directory will be automatically uploaded to pcloud after being processed.


Automation scripts description
------------
We currently have three versions of the automation script. Currently we will use version 3 where all the latest updates is done.


* version1 (`gql_client_proxy_v1.py`)  runs for a specified duration, which can be configured both within the script and through crontab. After completing the defined time period, the script terminates and is automatically restarted by cron

* version2 (`gql_client_proxy_v2.py`) runs continuously once started from crontab. It remains active without interruption. Uploads data to Pcloud

* version3 (`gql_client_proxy_v3.py`) Same as V2 but Uploads data to Digital Ocean.


Semi automation ( Improvement/ modification if needed after the data annotation is uploaded)
------------
Once the automated video processing is complete, the results are uploaded to pCloud. However, if certain scenes require further enhancements (e.g., adding a hand model, removing shadows), then add the necessary furhter processing arguments  inside the script `updatepcloudResults.py` and run the script. This will do the necessary modification to the current results, uploads and replaces with new results in the pcloud.
```py
python updatepcloudResults.py --folder_name <folder_name> --directory <directory> --folder_id <folder_id> 
```
Example usage:
```py
python updatepcloudResults.py --folder_name cm8ek9jj2003oqe0cfeqcou7s --directory /home/user/PREMIERETOOLV2 --folder_id 12345678
```
Features of the scripts:
------------
* Automation using crontab
* Send email to the receipient  if error any occurs and if the video processing is complete
* Handle mulitple video uploads from cms, stores them in queue, downloads and processes them one by one.
* All the logs store in a log file `gql_proxy.log`
* Upload the video after processing in required format
* Improve/modify on the current result by running `updatepcloudResults.py` if necessary. (Semi-automation)

