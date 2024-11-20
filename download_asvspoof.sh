wget -O asvspoof2021-df-data.zip https://zenodo.org/api/records/4835108/files-archive
mkdir -p egs/asvspoof2021/data
unzip asvspoof2021-df-data.zip -d egs/asvspoof2021/data
rm asvspoof2021-df-data.zip
cd egs/asvspoof2021/data
for file in *.tar; do
    tar -xvf "$file" -C ./
done
tar files if no longer needed
rm *.tar