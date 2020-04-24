# sent2vec_for_vs

sent2vec original src code 
https://github.com/epfml/sent2vec

hnswlib original src code 
https://github.com/nmslib/hnswlib

add command "test_nn": 
generate sentence embeddings from a TXT file, map them into an hnsw space, return top 3 ann sentences, query is first 500 sentences by default. (io path not modified-- modify add_points(), test_nn() in fasttext.cpp)(for changing hnsw parameters, modify hnsw_init() in fasttext.cpp)(to use your own distance function, modify InnerProduct() in hnswlib/space_ip.h)

add command "generate_s2v $model $input_file $output_file": 
input sentences.txt, output $sentence\t$embeddings
