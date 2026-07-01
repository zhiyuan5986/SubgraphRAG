# path=webqsp_Jun19-02:56:52
# python visualize_retriever_bce_scores.py \
#         -p $path/cpt.pth \
#         -d webqsp \
#         --data_dir . \
#         --output_dir $path/visualize_scores \
#         --device cuda:1 \

path=cwq_Jun30-09:03:26
python visualize_retriever_bce_scores.py \
        -p $path/cpt.pth \
        -d cwq \
        --data_dir . \
        --output_dir $path/visualize_scores \
        --device cuda:1 \