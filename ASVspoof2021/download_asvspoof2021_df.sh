wget -O asvspoof2021-df-data.zip https://zenodo.org/api/records/4835108/files-archive
mkdir data
unzip asvspoof2021-df-data.zip -d data
rm asvspoof2021-df-data.zip
cd data
for file in *.tar.gz; do
    tar -xvf "$file" -C ./
done
rm *.tar.gz