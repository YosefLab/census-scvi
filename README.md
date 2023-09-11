# census-scvi

## Instructions to set up in AWS

1. Launch an instance with the [Deep Learning Base GPU AMI](https://aws.amazon.com/releasenotes/aws-deep-learning-base-gpu-ami-ubuntu-20-04/)

2. Log into the instance and configure GPUs to run inside Docker containers
    ```
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

3. Pull the image:
   
   ```
   docker pull martinkim0/scvi-tools:py3.11-cu11-autotune-main
   ```

4. Mount the AWS drive to a directory
   ```
   sudo mkdir /data
   sudo mount /dev/mapper/name-of-drive /data
   ```

5. Clone this repository
   ```
   git clone https://github.com/YosefLab/census-scvi.git
   ```

6. Run the container with a mounted volume a