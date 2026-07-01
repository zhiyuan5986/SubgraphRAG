# dataset=webqsp
# python eval.py -d $dataset \
#                --path webqsp_Jun19-02:56:52/retrieval_result.pth \
#                --thres="-7" \
#                --visualize \
#                --thres_k 100

            #    --k_list '50,100,200,400,500,600,700,800,900,1000'

# dataset=cwq
# python eval.py -d $dataset \
#                --path cwq_Jun30-09:03:26/retrieval_result.pth \
#                --thres="-7" \
#                --visualize \
#                --thres_k 100

dataset=cwq
python eval.py -d $dataset \
               --path webqsp_Jun19-02:56:52/retrieval_result.pth \
