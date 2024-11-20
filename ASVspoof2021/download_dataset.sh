wget -O asvspoof2021-df-data.zip https://zenodo.org/api/records/4835108/files-archive
mkdir -p ASVspoof2021/data
unzip asvspoof2021-df-data.zip -d ASVspoof2021/data
rm asvspoof2021-df-data.zip
cd ASVspoof2021/data
for file in *.tar.gz; do
    tar -xvf "$file" -C ./
done
rm *.tar.gz