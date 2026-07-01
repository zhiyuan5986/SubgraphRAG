# dataset=webqsp
# python inference.py --path webqsp_Jun19-02:56:52/cpt.pth --max_K 500
# dataset=cwq
# python inference.py --path cwq_Jun30-09:03:26/cpt.pth --max_K 500
dataset=cwq
python inference.py --path webqsp_Jun19-02:56:52/cpt.pth --max_K 500 --dataset $dataset

