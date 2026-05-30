## Improved Collaborative Recommendation Framework Based on Enhanced Knowledge Representation Learning

The paper is available at .


## Requirements

* torch==2.5.1+cu121
torch-geometric==2.3.0
torchaudio==0.20.1+cu121
sentence-transformers==2.3.0
numpy==1.21.0
h5py==3.0.0
pandas==1.3.0


## How to Run

Due to file size limitations, movie and book-related data are uploaded to https://pan.baidu.com/s/1W7hGkwrDWXqntBipdaeKeg?pwd=en23

### ml-1m dataset

Step 1, run build_ckg.py generate ./ckg_ml1m/graph_edges.txt node_info.json train.txt valid.txt test.txt relaton_to_id.json

Step 2, run OSE.py generate ml_embeddings.h5

Step 3, run trainmovie.py generate ./mtrainmodel/neighbor_dict.pkl best_model.pt checkpoint.pt

step 4, run recommender.py generate ./mtrainmodel/test_results.json


### book dataset

Step 1, run build_ckg_book.py generate ./ckg_book/graph_edges.txt node_info.json train.txt valid.txt test.txt relaton_to_id.json

Step 2, run OSE.py generate book_embeddings.h5

Step 3, run trainbook.py generate ./btrainmodel/neighbor_dict.pkl best_model.pt checkpoint.pt

step 4, run recommender.py generate ./btrainmodel/test_results.json


### FB15K237 and WN18RR dataset 

Step 1, run preprocess.py/preprocess2.py generate ./FB15k237/node_info.json ./WN18RR/node_info.json

Step 2, run OSE.py generate FB_embeddings.h5 WN_embedding.h5

Step 3, run train.py generate ./trainmodel/neighbor_dict.pkl best_model.pt checkpoint.pt relaton_to_id.json

step 4, run test.py generate ./trainmodel/test_results.json

## Citation

If you find our paper or code repository helpful, please consider citing as follows:

```

```
