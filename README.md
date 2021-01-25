# patent-analysis

sudo docker build -f Dockerfile -t patent_analysis .

docker run -p 1234:8888 -p 6000-6100:6000-6100 -v ~/workspace/patent-analysis:/patent-analysis -t -d --name patent_analysis_session patent_analysis
