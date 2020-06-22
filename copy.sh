name=$1
server='AzureUser@ewerton.southcentralus.cloudapp.azure.com'
dir='/home/AzureUser/experimentos/'
#scp *.py AzureUser@13.84.184.250:/home/AzureUser/experimentos
scp $server:$dir*.csv ./
scp -r $server:$dir$name/ ./
scp $server:$dir$name.hdf5 ./


